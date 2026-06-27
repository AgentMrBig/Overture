"""
Overture — Contrastive Pre-trainer v3
======================================
Trains the complex token encoder to mirror Qwen3-8B's own similarity
geometry. Qwen3-8B is both teacher and embedding source — one model,
self-consistent geometry.

Key changes from v2:
  - Teacher: Qwen3-8B hidden states (4096-dim) instead of BGE-large (1024-dim)
  - Encoder input_dim: 4096 to match shared model architecture
  - No separate BGE download needed
  - Saves checkpoint compatible with OvertureFrame(shared_model=...)

Target: correlation > 0.85 with Qwen3-8B in complex space

Usage:
    python contrastive_pretrain_v3.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import random
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from token_encoder import DomainRegistry, TokenEncoder


# ─────────────────────────────────────────────────────────
#  QWEN3-8B EMBEDDER
#  Mean-pools last hidden states — same as ChatModel.get_embeddings()
# ─────────────────────────────────────────────────────────

class Qwen3Embedder:
    def __init__(self, device):
        self.device = device
        self.model  = None
        self.tok    = None

    def load(self):
        model_name = "Qwen/Qwen3-8B"
        print(f"  Loading Qwen3-8B (4-bit) for training...")
        t0 = time.perf_counter()

        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        bnb = BitsAndBytesConfig(
            load_in_4bit           = True,
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type    = "nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb,
            device_map          = "auto",
            trust_remote_code   = True,
            max_memory          = {0: "75GiB", 1: "75GiB", "cpu": "40GiB"},
        )
        print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    @torch.no_grad()
    def encode(self, texts: list, batch_size: int = 16) -> torch.Tensor:
        """Encode list of texts -> (N, 4096) normalized float tensor."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self.tok(
                batch,
                return_tensors  = "pt",
                truncation      = True,
                max_length      = 256,
                padding         = True,
            ).to(self.model.device)
            out    = self.model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[-1]                          # (B, seq, 4096)
            mask   = inputs['attention_mask'].unsqueeze(-1).float()
            emb    = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-8)  # (B, 4096)
            all_embs.append(emb.float().cpu())
        embs = torch.cat(all_embs, dim=0)
        return embs / embs.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# ─────────────────────────────────────────────────────────
#  CORPUS BUILDER
# ─────────────────────────────────────────────────────────

def build_corpus(target_size: int = 20000) -> list:
    from datasets import load_dataset
    corpus = []
    seen   = set()

    def add(s: str):
        s = s.strip()
        if len(s) < 20 or len(s) > 300 or s in seen:
            return
        if not any(c.isalpha() for c in s):
            return
        seen.add(s)
        corpus.append(s)

    per = target_size // 5
    sources = [
        ('fancyzhx/ag_news',    'train', 'text',     per * 2, 2),
        ('rajpurkar/squad',      'train', 'context',  per,     3),
        ('dair-ai/emotion',      'train', 'text',     per,     1),
        ('nyu-mll/multi_nli',    'train', 'premise',  per,     1),
        ('nyu-mll/multi_nli',    'train', 'hypothesis', per,   1),
    ]

    for ds_name, split, field, max_items, sents_per in sources:
        print(f"  {ds_name}...", end='', flush=True)
        try:
            ds = load_dataset(ds_name, split=split, streaming=True)
            count = 0
            for item in ds:
                text = item.get(field, '').replace('\\n', ' ').strip()
                if not text:
                    continue
                if sents_per == 1:
                    add(text[:250])
                else:
                    for s in text.split('.')[:sents_per]:
                        if len(s.strip()) > 25:
                            add(s.strip() + '.')
                count += 1
                if count >= max_items:
                    break
            print(f" {len(corpus):,}")
        except Exception as e:
            print(f" failed: {e}")

    random.shuffle(corpus)
    corpus = corpus[:target_size]
    print(f"  Corpus: {len(corpus):,} sentences")
    return corpus


# ─────────────────────────────────────────────────────────
#  PAIR DATASET
# ─────────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, n_pairs: int = 30000, seed: int = 42):
        super().__init__()
        self.embs = embeddings.cpu().float()
        n         = len(self.embs)
        rng       = np.random.RandomState(seed)

        print(f"  Building similarity matrix ({n} x {n})...")
        E = self.embs.numpy()
        chunk = 500
        sim   = np.zeros((n, n), dtype=np.float32)
        for i in range(0, n, chunk):
            ei = min(i + chunk, n)
            for j in range(0, n, chunk):
                ej = min(j + chunk, n)
                sim[i:ei, j:ej] = E[i:ei] @ E[j:ej].T

        # Stratified sampling across similarity range
        weights      = [3, 2, 1, 1, 1, 1, 1, 1, 2, 3]
        total_weight = sum(weights)
        targets      = {b: max(1, int(n_pairs * w / total_weight))
                        for b, w in enumerate(weights)}
        buckets      = {b: [] for b in range(10)}
        filled, attempts = 0, 0

        while filled < n_pairs and attempts < n_pairs * 200:
            i, j = rng.randint(0, n, 2)
            if i == j:
                attempts += 1
                continue
            if i > j:
                i, j = j, i
            s      = sim[i, j]
            bucket = min(int(s * 10), 9)
            if len(buckets[bucket]) < targets[bucket]:
                buckets[bucket].append((i, j))
                filled += 1
            attempts += 1

        self.pairs = []
        for b in buckets.values():
            self.pairs.extend(b)
        rng.shuffle(self.pairs)
        self.pairs = self.pairs[:n_pairs]
        self.sim   = sim

        sims = [sim[i, j] for i, j in self.pairs]
        print(f"  Pairs: {len(self.pairs):,}  range [{min(sims):.3f}, {max(sims):.3f}]  mean {np.mean(sims):.3f}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.embs[i], self.embs[j], torch.tensor(self.sim[i, j], dtype=torch.float32)


# ─────────────────────────────────────────────────────────
#  STUDENT
# ─────────────────────────────────────────────────────────

class StudentSimilarity(nn.Module):
    def __init__(self, encoder: TokenEncoder):
        super().__init__()
        self.encoder = encoder

    def encode_one(self, emb):
        return self.encoder(emb.unsqueeze(1)).squeeze(1)

    def forward(self, a, b):
        ta   = self.encode_one(a)
        tb   = self.encode_one(b)
        ma   = ta.abs().float()
        mb   = tb.abs().float()
        dot  = (ma * mb).sum(dim=-1)
        norm = ma.norm(dim=-1).clamp(min=1e-8) * mb.norm(dim=-1).clamp(min=1e-8)
        return dot / norm


# ─────────────────────────────────────────────────────────
#  COMBINED LOSS
# ─────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        neg_mask = (target < 0.25).float()
        pos_mask = (target > 0.75).float()
        neg_loss = (torch.clamp(pred - 0.25 + 0.25, min=0) * neg_mask).mean()
        pos_loss = (torch.clamp(0.75 - pred + 0.25, min=0) * pos_mask).mean()
        return mse_loss + 0.8 * (neg_loss + pos_loss), mse_loss, neg_loss + pos_loss


# ─────────────────────────────────────────────────────────
#  EVAL
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(student, embedder, sentences, device):
    student.eval()
    embs   = embedder.encode(sentences).to(device)
    tokens = student.encode_one(embs)
    n      = len(sentences)

    qwen_sims, our_sims = [], []
    for i in range(n):
        for j in range(i + 1, n):
            qs = (embs[i] @ embs[j]).item()
            ma = tokens[i].abs().float()
            mb = tokens[j].abs().float()
            os_ = (ma @ mb / (ma.norm().clamp(min=1e-8) * mb.norm().clamp(min=1e-8))).item()
            qwen_sims.append(qs)
            our_sims.append(os_)

    corr = np.corrcoef(qwen_sims, our_sims)[0, 1]
    student.train()
    return corr


# ─────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────

def train(student, loader, val_sentences, embedder, device,
          n_epochs=60, lr=2e-3, target_corr=0.85):

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=lr * 0.01, max_lr=lr,
        step_size_up=len(loader) * 2, mode='triangular2', cycle_momentum=False,
    )
    criterion = CombinedLoss()

    print(f"\n{'═'*66}")
    print(f"  Contrastive Pre-training v3 — Qwen3-8B teacher")
    print(f"  Target: >{target_corr:.0%} correlation  |  Max epochs: {n_epochs}")
    print(f"{'═'*66}")
    print(f"  {'Ep':>4}  {'Loss':>10}  {'MSE':>8}  {'Margin':>8}  {'Corr':>8}  {'Time':>6}")
    print(f"  {'─'*56}")

    best_corr, best_state = 0.0, None
    t_start = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        t_ep = time.perf_counter()
        total_l, mse_l, margin_l = [], [], []

        for a, b, target in loader:
            a, b, target = a.to(device), b.to(device), target.to(device)
            optimizer.zero_grad()
            pred = student(a, b)
            loss, mse, margin = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_l.append(loss.item())
            mse_l.append(mse.item())
            margin_l.append(margin.item())

        elapsed = time.perf_counter() - t_ep

        if epoch % 2 == 0 or epoch == 1:
            corr = evaluate(student, embedder, val_sentences, device)
            star = '★' if corr > best_corr else ' '
            if corr > best_corr:
                best_corr  = corr
                best_state = {k: v.clone() for k, v in student.encoder.state_dict().items()}
            print(f"  {epoch:>4}  {np.mean(total_l):>10.6f}  {np.mean(mse_l):>8.6f}  "
                  f"{np.mean(margin_l):>8.6f}  {corr:>8.4f}  {elapsed:>5.1f}s {star}")
            if corr >= target_corr:
                print(f"\n  Target {target_corr:.0%} reached at epoch {epoch}!")
                break
        else:
            print(f"  {epoch:>4}  {np.mean(total_l):>10.6f}  {np.mean(mse_l):>8.6f}  "
                  f"{np.mean(margin_l):>8.6f}  {'─':>8}  {elapsed:>5.1f}s")

    print(f"  {'─'*56}")
    print(f"  Best corr : {best_corr:.4f}  |  Time: {(time.perf_counter()-t_start)/60:.1f} min")
    print(f"{'═'*66}\n")
    return best_corr, best_state


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    TARGET_SIZE = 20000
    N_PAIRS     = 30000
    BATCH_SIZE  = 256
    N_EPOCHS    = 60
    LR          = 2e-3
    INPUT_DIM   = 4096   # Qwen3-8B hidden size

    # Load embedder
    embedder = Qwen3Embedder(device)
    embedder.load()

    # Build / load corpus
    cache_path = 'corpus_embeddings_v3.pt'
    if os.path.exists(cache_path):
        print(f"\nLoading cached embeddings...")
        cache  = torch.load(cache_path, weights_only=True)
        corpus = cache['corpus']
        embs   = cache['embeddings']
        print(f"  {len(corpus):,} sentences, shape {tuple(embs.shape)}")
    else:
        print(f"\nBuilding corpus ({TARGET_SIZE:,} sentences)...")
        corpus = build_corpus(TARGET_SIZE)
        print(f"\nEncoding with Qwen3-8B (batches of 16)...")
        t0   = time.perf_counter()
        embs = embedder.encode(corpus, batch_size=16)
        print(f"  Encoded {len(corpus):,} sentences in {time.perf_counter()-t0:.1f}s")
        torch.save({'embeddings': embs.cpu(), 'corpus': corpus}, cache_path)
        print(f"  Cached to {cache_path}")

    embs = embs.to(device)

    # Validation sentences
    val_sentences = [
        "The stars illuminate the darkness of space.",
        "Night is defined by the absence of sunlight.",
        "Happiness is a choice made moment by moment.",
        "Depression is a weight that colors everything grey.",
        "Science and art are two ways of understanding reality.",
        "Religion offers meaning where reason falls silent.",
        "A child's laughter is the purest sound in the world.",
        "Silence can be more powerful than any spoken word.",
        "The universe is indifferent to human suffering.",
        "Compassion is the recognition of shared vulnerability.",
        "Speed is the distance traveled divided by time elapsed.",
        "Patience is the ability to wait without losing peace.",
        "Fire destroys what took years to build.",
        "Water gives life to everything it touches.",
        "The algorithm processed millions of records per second.",
        "She wept quietly in the corner of the empty room.",
        "Justice delayed is justice denied.",
        "The market closed at an all-time high today.",
        "Wolves hunt in coordinated packs across vast territories.",
        "Democracy requires constant vigilance to survive.",
    ]

    # Build pair dataset
    print(f"\nBuilding pair dataset ({N_PAIRS:,} pairs)...")
    dataset = PairDataset(embs.cpu(), n_pairs=N_PAIRS, seed=42)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, pin_memory=(device.type == 'cuda'))

    # Build student — input_dim=4096 to match Qwen3-8B
    registry = DomainRegistry(d_model=64, k_sparse=16)
    registry.register('qwen', input_dim=INPUT_DIM)
    registry.to(device)
    student = StudentSimilarity(registry.encoders['qwen'])
    student.to(device)
    print(f"\nStudent parameters: {sum(p.numel() for p in student.parameters()):,}")

    # Baseline
    print(f"\nBaseline correlation (random weights):")
    baseline = evaluate(student, embedder, val_sentences, device)
    print(f"  {baseline:.4f}")

    # Train
    best_corr, best_state = train(
        student, loader, val_sentences, embedder, device,
        n_epochs=N_EPOCHS, lr=LR, target_corr=0.85,
    )

    # Load best
    registry.encoders['qwen'].load_state_dict(best_state)

    # Final eval
    final_corr = evaluate(student, embedder, val_sentences, device)
    print(f"Baseline : {baseline:.4f}")
    print(f"Final    : {final_corr:.4f}  (+{final_corr - baseline:.4f})")

    # Save
    out_path = 'qwen_encoder_pretrained.pt'
    torch.save({
        'encoder_state': registry.encoders['qwen'].state_dict(),
        'd_model'      : 64,
        'k_sparse'     : 16,
        'input_dim'    : INPUT_DIM,
        'base_model'   : 'Qwen/Qwen3-8B',
        'final_corr'   : final_corr,
        'corpus_size'  : len(corpus),
        'n_pairs'      : N_PAIRS,
    }, out_path)
    print(f"\nSaved: {out_path}")
