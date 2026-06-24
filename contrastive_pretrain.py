"""
Overture — Language Encoder Contrastive Pre-trainer
=====================================================
Trains the language domain encoder to preserve BGE-large's
semantic geometry in our 64-dim complex space.

Method: Knowledge Distillation via Similarity Matching
    Teacher : BGE-large (frozen, already understands language)
    Student : Our language DomainEncoder (being trained)

    For every pair of sentences (A, B):
        teacher_sim = cosine_similarity(BGE(A), BGE(B))
        student_sim = complex_similarity(Encoder(BGE(A)), Encoder(BGE(B)))
        loss = MSE(student_sim, teacher_sim)

    The encoder learns to reproduce BGE's distance geometry
    in our complex space. No labels needed — BGE is the label.

Training data: diverse sentences covering broad semantic range
    - Pulled from a built-in corpus of ~2000 varied sentences
    - Covers concrete/abstract, positive/negative, many domains
    - Pairs sampled to ensure coverage of full similarity range

After training:
    Structure survival correlation should jump from ~0.14 → >0.80
    Encoder checkpoint saved to: lang_encoder_pretrained.pt

Usage:
    python contrastive_pretrain.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from token_encoder import DomainRegistry, TokenEncoder


# ─────────────────────────────────────────────────────────
#  TRAINING CORPUS
#  Diverse sentences covering broad semantic territory.
#  Variety matters more than quantity here — we want
#  pairs that span the full range from 0.0 to 1.0 similarity.
# ─────────────────────────────────────────────────────────

CORPUS = [
    # Concrete physical world
    "The sun rises in the east every morning.",
    "Stars are massive balls of burning gas.",
    "Water flows downhill due to gravity.",
    "Trees absorb carbon dioxide from the air.",
    "Mountains form over millions of years.",
    "The ocean covers most of the Earth's surface.",
    "Fire requires oxygen, fuel, and heat.",
    "Ice melts when the temperature rises above zero.",
    "Rain falls from clouds when water vapor condenses.",
    "Wind is caused by differences in air pressure.",
    "Earthquakes occur at tectonic plate boundaries.",
    "Volcanoes release molten rock from deep underground.",
    "Deserts receive very little rainfall annually.",
    "Forests are home to countless species of life.",
    "Rivers carry sediment from mountains to the sea.",

    # Abstract concepts
    "Justice requires equal treatment under the law.",
    "Freedom means the ability to act without constraint.",
    "Truth is independent of what anyone believes.",
    "Beauty exists in the eye of the beholder.",
    "Time moves forward and cannot be reversed.",
    "Knowledge grows through observation and reasoning.",
    "Wisdom comes from reflecting on experience.",
    "Courage means acting despite the presence of fear.",
    "Love binds people together across all differences.",
    "Hope sustains people through difficulty and hardship.",
    "Grief is the natural response to profound loss.",
    "Curiosity drives exploration and discovery.",
    "Imagination creates possibilities that do not yet exist.",
    "Memory preserves the past within the present mind.",
    "Identity is shaped by experience and choice.",

    # Science and technology
    "Quantum mechanics describes the behavior of subatomic particles.",
    "Relativity shows that space and time are interconnected.",
    "Evolution occurs through natural selection over generations.",
    "DNA carries the genetic instructions for all living things.",
    "Entropy always increases in a closed system.",
    "Light travels at approximately three hundred thousand kilometers per second.",
    "Black holes are regions where gravity is so strong nothing escapes.",
    "Neurons transmit electrical signals throughout the nervous system.",
    "Algorithms are step-by-step procedures for solving problems.",
    "Machine learning finds patterns in large datasets automatically.",
    "Cryptography protects information through mathematical transformations.",
    "Semiconductors form the foundation of modern computing.",
    "Nuclear fusion powers the stars in our galaxy.",
    "Antibiotics target bacterial infections but not viruses.",
    "Climate change is driven by increasing greenhouse gas emissions.",

    # Human experience and emotion
    "Laughter is a universal expression of joy and amusement.",
    "Anger arises when expectations are violated or boundaries crossed.",
    "Fear protects us by alerting us to potential danger.",
    "Sadness accompanies loss and helps us process grief.",
    "Excitement builds when we anticipate something positive.",
    "Boredom signals that our minds need more stimulation.",
    "Guilt arises when we act against our own values.",
    "Pride follows achievement and effort well spent.",
    "Loneliness results from a lack of meaningful connection.",
    "Contentment is the quiet satisfaction of enough.",
    "Anxiety fills the mind with imagined future threats.",
    "Nostalgia colors past memories with warmth and longing.",
    "Jealousy emerges when we fear losing what we value.",
    "Gratitude opens us to recognizing what we have been given.",
    "Empathy allows us to feel what another person feels.",

    # Actions and behaviors
    "Running builds cardiovascular endurance over time.",
    "Reading expands vocabulary and deepens understanding.",
    "Building requires planning, materials, and skill.",
    "Destroying takes far less effort than creating.",
    "Teaching transfers knowledge from one mind to another.",
    "Learning requires attention, repetition, and reflection.",
    "Listening is more difficult than speaking for most people.",
    "Writing forces clarity of thought and expression.",
    "Cooking transforms raw ingredients into nourishment.",
    "Sleeping allows the brain to consolidate memories.",
    "Exercising releases endorphins that improve mood.",
    "Meditating trains attention and reduces mental noise.",
    "Collaborating produces outcomes beyond individual capacity.",
    "Competing drives performance but can damage relationships.",
    "Exploring expands the boundaries of the known world.",

    # Contrasting pairs — positive vs negative
    "The city is alive with energy and possibility.",
    "The city is exhausting and overwhelming to live in.",
    "Technology connects people across vast distances.",
    "Technology isolates people from genuine human contact.",
    "Growth requires leaving comfort zones behind.",
    "Growth can be painful and disorienting to experience.",
    "Change brings new opportunities and fresh beginnings.",
    "Change destroys what was familiar and beloved.",
    "Success comes from persistent effort and discipline.",
    "Success can corrupt character and distort values.",
    "Science reveals the hidden order of the universe.",
    "Science cannot answer the deepest questions of meaning.",
    "Cities are centers of culture, commerce, and innovation.",
    "Cities breed inequality, pollution, and social isolation.",
    "The future holds tremendous promise for humanity.",
    "The future is uncertain and filled with existential risk.",

    # Highly similar near-duplicate pairs
    "The dog barked loudly at the passing stranger.",
    "The canine growled noisily at the unknown visitor.",
    "She smiled warmly at everyone who entered the room.",
    "She grinned cheerfully at each person who came in.",
    "The economy grew rapidly in the first quarter.",
    "Economic growth accelerated significantly in Q1.",
    "He finished the book in a single afternoon.",
    "He completed reading the novel within one sitting.",
    "The storm passed and the sun emerged from behind the clouds.",
    "After the rain ended the sky cleared and brightened.",

    # Completely unrelated — low similarity targets
    "Purple is a combination of red and blue light.",
    "The stock market closed higher on strong earnings.",
    "Elephants communicate through low-frequency vibrations.",
    "The treaty was signed after months of negotiation.",
    "Chocolate is derived from the cacao tree.",
    "Submarines operate at great depths underwater.",
    "The violin produces sound through vibrating strings.",
    "Democracy requires an informed and engaged citizenry.",
    "Coffee contains caffeine which stimulates the nervous system.",
    "Bridges must withstand wind, traffic, and seismic forces.",
    "The painting sold at auction for millions of dollars.",
    "Bacteria can develop resistance to antibiotics over time.",
    "Skiing requires balance, technique, and physical fitness.",
    "The constitution protects fundamental individual rights.",
    "Photosynthesis converts sunlight into chemical energy.",

    # Cross-domain bridges — moderate similarity
    "Music and mathematics share deep structural relationships.",
    "Language shapes the way we perceive and understand reality.",
    "Art expresses what cannot be captured in words alone.",
    "Architecture embodies the values of the culture that built it.",
    "Sport reveals character under pressure and adversity.",
    "Philosophy questions assumptions that science takes for granted.",
    "History repeats itself when its lessons are forgotten.",
    "Economics studies how people allocate scarce resources.",
    "Psychology examines the hidden forces that drive behavior.",
    "Ecology studies the relationships between living organisms.",
]


# ─────────────────────────────────────────────────────────
#  DATASET
#  Generates pairs of sentences with their BGE similarity
#  as the target. Samples pairs to cover full range.
# ─────────────────────────────────────────────────────────

class SimilarityPairDataset(Dataset):
    """
    Dataset of (embedding_A, embedding_B, target_similarity) triples.

    Pre-computes all BGE embeddings once, then samples pairs
    during training. This avoids running BGE on every batch.

    Args:
        embeddings  : (N, 1024) BGE embeddings of all corpus sentences
        n_pairs     : number of pairs to generate per epoch
        seed        : random seed
    """
    def __init__(
        self,
        embeddings : torch.Tensor,
        n_pairs    : int = 8000,
        seed       : int = 42,
    ):
        super().__init__()
        self.embeddings = embeddings.cpu()
        self.n_pairs    = n_pairs
        self.n          = len(embeddings)
        rng = np.random.RandomState(seed)

        # Pre-compute all pairwise similarities (small enough to fit in memory)
        print(f"  Pre-computing {self.n}×{self.n} similarity matrix...")
        E = embeddings.float().cpu().numpy()
        # Normalize for cosine similarity
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E_norm = E / (norms + 1e-8)
        self.sim_matrix = (E_norm @ E_norm.T).astype(np.float32)

        # Sample pairs — stratified by similarity bucket
        # to ensure coverage of full [0, 1] range
        self.pairs = self._sample_stratified(rng, n_pairs)
        print(f"  Sampled {len(self.pairs)} pairs")
        sims = [self.sim_matrix[i, j] for i, j in self.pairs]
        print(f"  Similarity range: {min(sims):.3f} – {max(sims):.3f}  "
              f"mean: {np.mean(sims):.3f}")

    def _sample_stratified(self, rng, n_pairs):
        """Sample pairs uniformly across similarity buckets."""
        buckets = np.linspace(0, 1, 11)  # 10 buckets
        pairs = []
        per_bucket = n_pairs // 10

        # Build bucket lookup
        bucket_pairs = {i: [] for i in range(10)}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                sim = self.sim_matrix[i, j]
                bucket = min(int(sim * 10), 9)
                bucket_pairs[bucket].append((i, j))

        for bucket_idx, bucket_list in bucket_pairs.items():
            if not bucket_list:
                continue
            chosen = rng.choice(
                len(bucket_list),
                size=min(per_bucket, len(bucket_list)),
                replace=False
            )
            for idx in chosen:
                pairs.append(bucket_list[idx])

        rng.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        emb_a = self.embeddings[i]
        emb_b = self.embeddings[j]
        target = torch.tensor(self.sim_matrix[i, j], dtype=torch.float32)
        return emb_a, emb_b, target


# ─────────────────────────────────────────────────────────
#  STUDENT MODEL WRAPPER
#  Wraps our language DomainEncoder with a similarity
#  computation for use in the distillation loss.
# ─────────────────────────────────────────────────────────

class StudentSimilarity(nn.Module):
    """
    Wraps the language domain encoder.
    Given two embeddings, returns their similarity in our complex space.
    """
    def __init__(self, encoder: TokenEncoder):
        super().__init__()
        self.encoder = encoder

    def encode_one(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (batch, 1024) → token: (batch, 64) complex"""
        emb_3d = emb.unsqueeze(1)               # (B, 1, 1024)
        token  = self.encoder(emb_3d)            # (B, 1, 64) complex
        return token.squeeze(1)                  # (B, 64) complex

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """
        emb_a, emb_b : (batch, 1024) BGE embeddings
        returns       : (batch,) similarity scores in [0, 1]
        """
        tok_a = self.encode_one(emb_a)   # (B, 64) complex
        tok_b = self.encode_one(emb_b)   # (B, 64) complex

        # Magnitude-weighted cosine similarity in complex space
        mag_a = tok_a.abs()              # (B, 64) real
        mag_b = tok_b.abs()

        dot   = (mag_a * mag_b).sum(dim=-1)                    # (B,)
        norm_a = mag_a.norm(dim=-1).clamp(min=1e-8)            # (B,)
        norm_b = mag_b.norm(dim=-1).clamp(min=1e-8)

        sim = dot / (norm_a * norm_b)   # (B,) in [0, 1] since magnitudes >= 0
        return sim


# ─────────────────────────────────────────────────────────
#  EVALUATION UTILITY
#  Measures structure survival correlation on a test set
# ─────────────────────────────────────────────────────────

def evaluate_correlation(student, bge_model, test_sentences, device):
    """
    Measure how well our encoder preserves BGE's similarity structure.
    Returns Pearson correlation between BGE similarities and our similarities.
    """
    student.eval()
    with torch.no_grad():
        # Get BGE embeddings
        embs = bge_model.encode(
            test_sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=device,
            show_progress_bar=False,
        )

        # Get our tokens
        embs_3d = embs.unsqueeze(1)
        tokens  = student.encoder(embs_3d).squeeze(1)

        # Compute pairwise similarities in both spaces
        n = len(test_sentences)
        bge_sims, our_sims = [], []

        E = embs.float()
        norms = E.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        E_norm = E / norms

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
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────

def train_contrastive(
    student      : StudentSimilarity,
    train_loader : DataLoader,
    val_sentences: list,
    bge_model,
    device       : torch.device,
    n_epochs     : int   = 40,
    lr           : float = 1e-3,
    target_corr  : float = 0.85,
):
    optimizer = optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )
    criterion = nn.MSELoss()

    print(f"\n{'═'*62}")
    print(f"  Contrastive Pre-training — Language Encoder")
    print(f"  Target correlation: >{target_corr:.0%}  |  Max epochs: {n_epochs}")
    print(f"{'═'*62}")
    print(f"  {'Epoch':>5}  {'Loss':>10}  {'Corr':>8}  {'LR':>10}  {'Time':>6}")
    print(f"  {'─'*50}")

    best_corr      = 0.0
    best_state     = None
    t_train_start  = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        t_epoch = time.perf_counter()
        student.train()
        epoch_losses = []

        for emb_a, emb_b, target in train_loader:
            emb_a  = emb_a.to(device)
            emb_b  = emb_b.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = student(emb_a, emb_b)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()

        avg_loss = np.mean(epoch_losses)
        cur_lr   = optimizer.param_groups[0]['lr']
        elapsed  = time.perf_counter() - t_epoch

        # Evaluate every 2 epochs
        if epoch % 2 == 0 or epoch == 1:
            corr = evaluate_correlation(student, bge_model, val_sentences, device)
            improved = '★' if corr > best_corr else ' '

            if corr > best_corr:
                best_corr  = corr
                best_state = {k: v.clone() for k, v in student.encoder.state_dict().items()}

            print(f"  {epoch:>5}  {avg_loss:>10.6f}  {corr:>7.4f}  "
                  f"{cur_lr:>10.6f}  {elapsed:>5.1f}s {improved}")

            # Early exit if target reached
            if corr >= target_corr:
                print(f"\n  Target correlation {target_corr:.0%} reached at epoch {epoch}!")
                break
        else:
            print(f"  {epoch:>5}  {avg_loss:>10.6f}  {'─':>8}  "
                  f"{cur_lr:>10.6f}  {elapsed:>5.1f}s")

    total_time = time.perf_counter() - t_train_start
    print(f"  {'─'*50}")
    print(f"  Best correlation  : {best_corr:.4f}")
    print(f"  Total train time  : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'═'*62}\n")

    return best_corr, best_state


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load BGE-large (teacher) ──
    print(f"\nLoading BGE-large (teacher)...")
    t0 = time.perf_counter()
    bge = SentenceTransformer('BAAI/bge-large-en-v1.5')
    bge.to(device)
    print(f"Loaded in {time.perf_counter()-t0:.2f}s")

    # ── Pre-compute BGE embeddings for entire corpus ──
    print(f"\nEncoding {len(CORPUS)} corpus sentences with BGE-large...")
    t0 = time.perf_counter()
    with torch.no_grad():
        corpus_embeddings = bge.encode(
            CORPUS,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=device,
            show_progress_bar=True,
            batch_size=32,
        )
    print(f"Encoded in {time.perf_counter()-t0:.2f}s")
    print(f"Embedding shape: {tuple(corpus_embeddings.shape)}")

    # ── Build dataset ──
    print(f"\nBuilding pair dataset...")
    dataset = SimilarityPairDataset(
        embeddings = corpus_embeddings,
        n_pairs    = 10000,
        seed       = 42,
    )
    loader = DataLoader(
        dataset,
        batch_size  = 128,
        shuffle     = True,
        num_workers = 0,
    )

    # ── Build student (our language encoder) ──
    registry = DomainRegistry(d_model=64, k_sparse=16)
    registry.register('lang', input_dim=1024)
    registry.to(device)

    student = StudentSimilarity(registry.encoders['lang'])
    student.to(device)

    params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {params:,}")

    # ── Validation sentences — different from corpus ──
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
    ]

    # ── Baseline correlation before training ──
    print(f"\nBaseline correlation (before training):")
    baseline = evaluate_correlation(student, bge, val_sentences, device)
    print(f"  Correlation: {baseline:.4f}")

    # ── Train ──
    best_corr, best_state = train_contrastive(
        student       = student,
        train_loader  = loader,
        val_sentences = val_sentences,
        bge_model     = bge,
        device        = device,
        n_epochs      = 50,
        lr            = 1e-3,
        target_corr   = 0.85,
    )

    # ── Load best weights back into encoder ──
    registry.encoders['lang'].load_state_dict(best_state)

    # ── Final evaluation ──
    print(f"Final evaluation on validation sentences:")
    final_corr = evaluate_correlation(student, bge, val_sentences, device)
    print(f"  Baseline correlation : {baseline:.4f}")
    print(f"  Final correlation    : {final_corr:.4f}")
    print(f"  Improvement          : +{final_corr - baseline:.4f}")

    # ── Save encoder weights ──
    save_path = 'lang_encoder_pretrained.pt'
    torch.save({
        'encoder_state'  : registry.encoders['lang'].state_dict(),
        'd_model'        : 64,
        'k_sparse'       : 16,
        'input_dim'      : 1024,
        'bge_model'      : 'BAAI/bge-large-en-v1.5',
        'final_corr'     : final_corr,
    }, save_path)
    print(f"\nEncoder saved to: {save_path}")
    print(f"\nLanguage encoder is ready.")
    print(f"Next: build the REPL and load this checkpoint.")