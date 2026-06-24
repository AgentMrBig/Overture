"""
Overture — Language Encoder Contrastive Pre-trainer v2
=======================================================
Upgraded version with:
    - 40,000+ real sentences from diverse HuggingFace datasets
    - True dissimilar pairs (sim < 0.15) explicitly sampled
    - Cyclic LR — keeps exploring, avoids plateau
    - Combined loss: MSE + margin loss
      MSE   : match BGE's exact similarity scores
      Margin: push dissimilar pairs apart explicitly
    - Larger batch size for better gradient estimates
    - More frequent evaluation

Datasets used (all free, streaming):
    - ag_news          : news headlines (4 topics)
    - squad            : Wikipedia passages
    - emotion          : emotional text
    - yelp_polarity    : reviews
    - multi_nli        : sentence pairs (already diverse)

Target: correlation > 0.85 with BGE-large in our complex space

Usage:
    pip install datasets
    python contrastive_pretrain_v2.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import random
import os
from sentence_transformers import SentenceTransformer
from token_encoder import DomainRegistry, TokenEncoder


# ─────────────────────────────────────────────────────────
#  CORPUS BUILDER
#  Pulls sentences from HuggingFace datasets via streaming.
#  No full download — takes only what we need.
# ─────────────────────────────────────────────────────────

def build_corpus(target_size: int = 40000) -> list:
    """
    Pulls diverse sentences from multiple HF datasets.
    Uses streaming so we never download the full dataset.

    Returns a list of clean, diverse sentences.
    """
    from datasets import load_dataset

    corpus = []
    seen   = set()

    def add_sentence(s: str):
        s = s.strip()
        # Basic quality filters
        if len(s) < 20 or len(s) > 300:
            return
        if s in seen:
            return
        if not any(c.isalpha() for c in s):
            return
        seen.add(s)
        corpus.append(s)

    per_source = target_size // 5

    sources = [
        # (dataset_name, split, text_field, max_items, sentences_per_item)
        ('fancyzhx/ag_news',          'train', 'text',      per_source * 2, 2),
        ('rajpurkar/squad',            'train', 'context',   per_source,     3),
        ('dair-ai/emotion',            'train', 'text',      per_source,     1),
        ('fancyzhx/yelp_polarity',     'train', 'text',      per_source // 2, 2),
        ('nyu-mll/multi_nli',          'train', 'premise',   per_source // 2, 1),
        ('nyu-mll/multi_nli',          'train', 'hypothesis',per_source // 2, 1),
        ('SetFit/sst2',                'train', 'text',      per_source,     1),
        ('mteb/stsbenchmark-sts',      'test',  'sentence1', per_source // 4, 1),
        ('mteb/stsbenchmark-sts',      'test',  'sentence2', per_source // 4, 1),
    ]

    for ds_name, split, field, max_items, sents_per in sources:
        print(f"  Loading {ds_name} [{field}]...")
        try:
            ds = load_dataset(ds_name, split=split, streaming=True)
            count = 0
            for item in ds:
                text = item.get(field, '')
                if not text:
                    continue
                text = text.replace('\\n', ' ').strip()
                if sents_per == 1:
                    add_sentence(text[:250])
                else:
                    parts = [s.strip() for s in text.split('.') if len(s.strip()) > 25]
                    for s in parts[:sents_per]:
                        add_sentence(s + '.')
                count += 1
                if count >= max_items:
                    break
            print(f"    corpus size now: {len(corpus):,}")
        except Exception as e:
            print(f"    {ds_name} failed: {e}")

    # Shuffle and trim
    random.shuffle(corpus)
    corpus = corpus[:target_size]
    print(f"\n  Final corpus size: {len(corpus)} sentences")
    return corpus


# ─────────────────────────────────────────────────────────
#  DATASET WITH STRATIFIED + HARD NEGATIVE SAMPLING
# ─────────────────────────────────────────────────────────

class StratifiedPairDataset(Dataset):
    """
    Generates pairs stratified across the full similarity range.
    Explicitly includes hard negatives (sim < 0.15) and
    hard positives (sim > 0.85) to anchor both ends of the scale.

    Args:
        embeddings   : (N, 1024) normalized BGE embeddings
        n_pairs      : total pairs per epoch
        hard_neg_frac: fraction of pairs that are hard negatives
        seed         : random seed
    """
    def __init__(
        self,
        embeddings    : torch.Tensor,
        n_pairs       : int   = 50000,
        hard_neg_frac : float = 0.2,
        seed          : int   = 42,
    ):
        super().__init__()
        self.embeddings = embeddings.cpu().float()
        self.n          = len(embeddings)
        self.n_pairs    = n_pairs
        rng = np.random.RandomState(seed)

        # Compute similarity matrix in chunks to save memory
        print(f"  Computing similarity matrix for {self.n} sentences...")
        t0 = time.perf_counter()
        E = self.embeddings.numpy()
        # Already normalized, so sim = dot product
        # Do in chunks to avoid OOM
        chunk = 500
        self.sim_matrix = np.zeros((self.n, self.n), dtype=np.float32)
        for i in range(0, self.n, chunk):
            end_i = min(i + chunk, self.n)
            for j in range(0, self.n, chunk):
                end_j = min(j + chunk, self.n)
                self.sim_matrix[i:end_i, j:end_j] = E[i:end_i] @ E[j:end_j].T
        print(f"  Similarity matrix computed in {time.perf_counter()-t0:.1f}s")

        # Sample pairs on the fly — no need to enumerate all 800M pairs
        # Strategy: sample random pairs, bin by similarity, keep until each
        # bucket is full. Oversample extremes for better coverage.
        print(f"  Stratifying pairs (sampling strategy — memory efficient)...")

        # Target counts per bucket with extreme overrepresentation
        weights     = [3,2,1,1,1,1,1,1,2,3]
        total_weight= sum(weights)
        targets     = {b: max(1, int(n_pairs * w / total_weight))
                       for b, w in enumerate(weights)}
        buckets     = {b: [] for b in range(10)}
        filled      = 0
        attempts    = 0
        max_attempts= n_pairs * 200  # safety limit

        while filled < n_pairs and attempts < max_attempts:
            # Sample a random pair
            i = rng.randint(0, self.n)
            j = rng.randint(0, self.n)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            sim    = self.sim_matrix[i, j]
            bucket = min(int(sim * 10), 9)

            if len(buckets[bucket]) < targets[bucket]:
                buckets[bucket].append((i, j))
                filled += 1

            attempts += 1

        # Print bucket distribution
        print(f"  Bucket distribution:")
        for b, pairs in buckets.items():
            print(f"    [{b/10:.1f}-{(b+1)/10:.1f}]: {len(pairs):>8,} pairs")

        self.pairs = []
        for b in buckets.values():
            self.pairs.extend(b)
        rng.shuffle(self.pairs)
        self.pairs = self.pairs[:n_pairs]

        sims = [self.sim_matrix[i, j] for i, j in self.pairs]
        print(f"\n  Final pairs: {len(self.pairs):,}")
        print(f"  Similarity range: {min(sims):.3f} – {max(sims):.3f}  "
              f"mean: {np.mean(sims):.3f}")
        print(f"  Hard negatives (<0.2): "
              f"{sum(1 for s in sims if s < 0.2):,} pairs")
        print(f"  Hard positives (>0.8): "
              f"{sum(1 for s in sims if s > 0.8):,} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return (
            self.embeddings[i],
            self.embeddings[j],
            torch.tensor(self.sim_matrix[i, j], dtype=torch.float32)
        )


# ─────────────────────────────────────────────────────────
#  STUDENT MODEL
# ─────────────────────────────────────────────────────────

class StudentSimilarity(nn.Module):
    def __init__(self, encoder: TokenEncoder):
        super().__init__()
        self.encoder = encoder

    def encode_one(self, emb: torch.Tensor) -> torch.Tensor:
        emb_3d = emb.unsqueeze(1)
        token  = self.encoder(emb_3d)
        return token.squeeze(1)

    def forward(self, emb_a, emb_b):
        tok_a = self.encode_one(emb_a)
        tok_b = self.encode_one(emb_b)
        mag_a = tok_a.abs().float()
        mag_b = tok_b.abs().float()
        dot   = (mag_a * mag_b).sum(dim=-1)
        norm  = mag_a.norm(dim=-1).clamp(min=1e-8) * mag_b.norm(dim=-1).clamp(min=1e-8)
        return dot / norm


# ─────────────────────────────────────────────────────────
#  COMBINED LOSS
#  MSE:    match exact similarity scores
#  Margin: push dissimilar pairs below threshold,
#          pull similar pairs above threshold
# ─────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    MSE loss + margin ranking loss.

    MSE encourages exact similarity matching.
    Margin explicitly penalizes:
        - dissimilar pairs (target < neg_threshold) that are too close
        - similar pairs (target > pos_threshold) that are too far apart

    Args:
        mse_weight    : weight for MSE component
        margin_weight : weight for margin component
        neg_threshold : pairs below this are "negatives"
        pos_threshold : pairs above this are "positives"
        margin        : minimum required separation
    """
    def __init__(
        self,
        mse_weight    : float = 1.0,
        margin_weight : float = 0.5,
        neg_threshold : float = 0.3,
        pos_threshold : float = 0.7,
        margin        : float = 0.3,
    ):
        super().__init__()
        self.mse_w     = mse_weight
        self.margin_w  = margin_weight
        self.neg_thr   = neg_threshold
        self.pos_thr   = pos_threshold
        self.margin    = margin
        self.mse       = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MSE component
        mse_loss = self.mse(pred, target)

        # Margin component
        # For negative pairs: max(0, pred - neg_threshold + margin)
        neg_mask   = (target < self.neg_thr).float()
        neg_loss   = (torch.clamp(pred - self.neg_thr + self.margin, min=0) * neg_mask).mean()

        # For positive pairs: max(0, pos_threshold - pred + margin)
        pos_mask   = (target > self.pos_thr).float()
        pos_loss   = (torch.clamp(self.pos_thr - pred + self.margin, min=0) * pos_mask).mean()

        margin_loss = neg_loss + pos_loss

        return self.mse_w * mse_loss + self.margin_w * margin_loss, mse_loss, margin_loss


# ─────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────

def evaluate_correlation(student, bge_model, test_sentences, device):
    student.eval()
    with torch.no_grad():
        embs = bge_model.encode(
            test_sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=device,
            show_progress_bar=False,
        )
        embs_3d = embs.unsqueeze(1)
        tokens  = student.encoder(embs_3d).squeeze(1)

        n = len(test_sentences)
        bge_sims, our_sims = [], []
        E_norm = embs.float()

        for i in range(n):
            for j in range(i+1, n):
                bge_sim = (E_norm[i] @ E_norm[j]).item()
                mag_i = tokens[i].abs().float()
                mag_j = tokens[j].abs().float()
                our_sim = (mag_i @ mag_j / (
                    mag_i.norm().clamp(min=1e-8) *
                    mag_j.norm().clamp(min=1e-8)
                )).item()
                bge_sims.append(bge_sim)
                our_sims.append(our_sim)

    corr = np.corrcoef(bge_sims, our_sims)[0, 1]
    student.train()
    return corr


# ─────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────

def train(
    student,
    loader,
    val_sentences,
    bge_model,
    device,
    n_epochs    = 60,
    lr          = 2e-3,
    target_corr = 0.85,
):
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

    # Cyclic LR — keeps exploring, avoids plateau
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr    = lr * 0.01,
        max_lr     = lr,
        step_size_up = len(loader) * 2,
        mode       = 'triangular2',
        cycle_momentum = False,
    )

    criterion = CombinedLoss(
        mse_weight    = 1.0,
        margin_weight = 0.8,
        neg_threshold = 0.25,
        pos_threshold = 0.75,
        margin        = 0.25,
    )

    print(f"\n{'═'*66}")
    print(f"  Contrastive Pre-training v2 — Language Encoder")
    print(f"  Target: >{target_corr:.0%} correlation  |  Max epochs: {n_epochs}")
    print(f"  Loss: MSE + Margin  |  Optimizer: AdamW + CyclicLR")
    print(f"{'═'*66}")
    print(f"  {'Ep':>4}  {'Total':>10}  {'MSE':>8}  {'Margin':>8}  "
          f"{'Corr':>8}  {'Time':>6}")
    print(f"  {'─'*56}")

    best_corr  = 0.0
    best_state = None
    t_start    = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        t_ep = time.perf_counter()
        student.train()
        total_losses, mse_losses, margin_losses = [], [], []

        for emb_a, emb_b, target in loader:
            emb_a  = emb_a.to(device)
            emb_b  = emb_b.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = student(emb_a, emb_b)
            total_loss, mse_loss, margin_loss = criterion(pred, target)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_losses.append(total_loss.item())
            mse_losses.append(mse_loss.item())
            margin_losses.append(margin_loss.item())

        elapsed = time.perf_counter() - t_ep

        # Evaluate every 2 epochs
        if epoch % 2 == 0 or epoch == 1:
            corr = evaluate_correlation(student, bge_model, val_sentences, device)
            improved = '★' if corr > best_corr else ' '

            if corr > best_corr:
                best_corr  = corr
                best_state = {k: v.clone()
                              for k, v in student.encoder.state_dict().items()}

            print(f"  {epoch:>4}  {np.mean(total_losses):>10.6f}  "
                  f"{np.mean(mse_losses):>8.6f}  {np.mean(margin_losses):>8.6f}  "
                  f"{corr:>8.4f}  {elapsed:>5.1f}s {improved}")

            if corr >= target_corr:
                print(f"\n  Target {target_corr:.0%} reached at epoch {epoch}!")
                break
        else:
            print(f"  {epoch:>4}  {np.mean(total_losses):>10.6f}  "
                  f"{np.mean(mse_losses):>8.6f}  {np.mean(margin_losses):>8.6f}  "
                  f"{'─':>8}  {elapsed:>5.1f}s")

    total_time = time.perf_counter() - t_start
    print(f"  {'─'*56}")
    print(f"  Best correlation : {best_corr:.4f}")
    print(f"  Total time       : {total_time:.1f}s ({total_time/60:.1f} min)")
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

    TARGET_SIZE = 40000
    N_PAIRS     = 50000
    BATCH_SIZE  = 256
    N_EPOCHS    = 60
    LR          = 2e-3

    # ── Load BGE ──
    print(f"\nLoading BGE-large...")
    t0 = time.perf_counter()
    bge = SentenceTransformer('BAAI/bge-large-en-v1.5')
    bge.to(device)
    print(f"Loaded in {time.perf_counter()-t0:.2f}s")

    # ── Build corpus ──
    print(f"\nBuilding {TARGET_SIZE:,} sentence corpus from HuggingFace datasets...")
    corpus = build_corpus(TARGET_SIZE)

    # ── Encode with BGE in batches (with cache) ──
    cache_path = 'corpus_embeddings.pt'
    if os.path.exists(cache_path):
        print(f"\nLoading cached embeddings from {cache_path}...")
        cache = torch.load(cache_path, weights_only=True)
        corpus_embeddings = cache['embeddings'].to(device)
        cached_corpus = cache['corpus']
        if len(cached_corpus) == len(corpus):
            print(f"Cache hit — skipping BGE encoding")
        else:
            print(f"Cache size mismatch, re-encoding...")
            corpus_embeddings = None
    else:
        corpus_embeddings = None

    if corpus_embeddings is None:
        print(f"\nEncoding {len(corpus):,} sentences with BGE-large...")
        print(f"(This is the slow step — BGE needs to process everything once)")
        t0 = time.perf_counter()
        corpus_embeddings = bge.encode(
            corpus,
            convert_to_tensor    = True,
            normalize_embeddings = True,
            device               = device,
            show_progress_bar    = True,
            batch_size           = 64,
        )
        print(f"Encoded in {time.perf_counter()-t0:.1f}s")
        # Save cache
        torch.save({'embeddings': corpus_embeddings.cpu(), 'corpus': corpus}, cache_path)
        print(f"Embeddings cached to {cache_path}")

    print(f"Embedding shape: {tuple(corpus_embeddings.shape)}")

    # ── Build pair dataset ──
    print(f"\nBuilding stratified pair dataset ({N_PAIRS:,} pairs)...")
    dataset = StratifiedPairDataset(
        embeddings = corpus_embeddings,
        n_pairs    = N_PAIRS,
        seed       = 42,
    )
    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = True if device.type == 'cuda' else False,
    )

    # ── Build student ──
    registry = DomainRegistry(d_model=64, k_sparse=16)
    registry.register('lang', input_dim=1024)
    registry.to(device)
    student = StudentSimilarity(registry.encoders['lang'])
    student.to(device)
    print(f"\nStudent parameters: {sum(p.numel() for p in student.parameters()):,}")

    # ── Validation sentences ──
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

    # ── Baseline ──
    print(f"\nBaseline correlation before training:")
    baseline = evaluate_correlation(student, bge, val_sentences, device)
    print(f"  {baseline:.4f}")

    # ── Train ──
    best_corr, best_state = train(
        student       = student,
        loader        = loader,
        val_sentences = val_sentences,
        bge_model     = bge,
        device        = device,
        n_epochs      = N_EPOCHS,
        lr            = LR,
        target_corr   = 0.85,
    )

    # ── Load best weights ──
    registry.encoders['lang'].load_state_dict(best_state)

    # ── Final eval ──
    print(f"Final evaluation:")
    final_corr = evaluate_correlation(student, bge, val_sentences, device)
    print(f"  Baseline   : {baseline:.4f}")
    print(f"  Final      : {final_corr:.4f}")
    print(f"  Improvement: +{final_corr - baseline:.4f}")

    # ── Save ──
    torch.save({
        'encoder_state': registry.encoders['lang'].state_dict(),
        'd_model'      : 64,
        'k_sparse'     : 16,
        'input_dim'    : 1024,
        'bge_model'    : 'BAAI/bge-large-en-v1.5',
        'final_corr'   : final_corr,
        'corpus_size'  : len(corpus),
        'n_pairs'      : N_PAIRS,
    }, 'lang_encoder_pretrained.pt')
    print(f"\nSaved: lang_encoder_pretrained.pt")
    print(f"Language encoder ready for REPL.")