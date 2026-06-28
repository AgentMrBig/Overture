"""
Overture — Contrastive Pre-trainer v4
======================================
Trains the complex token encoder to mirror BGE-small-en-v1.5 similarity
geometry. BGE is a dedicated embedding model — its cosine similarities
have real variance and are a proper teacher signal.

Key changes from v3:
  - Teacher: BAAI/bge-small-en-v1.5 (768-dim) — real embedding model
  - INPUT_DIM: 768 to match BGE output
  - No BitsAndBytes, no 4-bit — BGE is tiny (33M params), runs in fp32
  - Cache file renamed to corpus_embeddings_v4.pt
  - Fixed best_state bug: student.state_dict() not student.encoder.state_dict()

Target: correlation > 0.85 with BGE teacher in complex space

Usage:
    python contrastive_pretrain_v4.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import random
import os
from transformers import AutoTokenizer, AutoModel
from token_encoder import DomainRegistry, TokenEncoder


# ─────────────────────────────────────────────────────────
#  BGE EMBEDDER — proper sentence embedding teacher
#  BAAI/bge-small-en-v1.5: 33M params, 768-dim, real cosine variance
# ─────────────────────────────────────────────────────────

class BGEEmbedder:
    def __init__(self, device):
        self.device = device
        self.model  = None
        self.tok    = None

    def load(self):
        model_name = "BAAI/bge-small-en-v1.5"
        print(f"  Loading BGE-small-en-v1.5...")
        t0 = time.perf_counter()
        self.tok   = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    @torch.no_grad()
    def encode(self, texts: list, batch_size: int = 64) -> torch.Tensor:
        """Encode list of texts -> (N, 768) CLS-pooled normalized float tensor."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch  = texts[i: i + batch_size]
            inputs = self.tok(
                batch,
                return_tensors = "pt",
                truncation     = True,
                max_length     = 256,
                padding        = True,
            ).to(self.device)
            out = self.model(**inputs)
            # BGE uses CLS token as sentence embedding
            emb = out.last_hidden_state[:, 0, :]   # (B, 768)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            all_embs.append(emb.float().cpu())
        return torch.cat(all_embs, dim=0)


# ─────────────────────────────────────────────────────────
#  CORPUS BUILDER
# ─────────────────────────────────────────────────────────

def build_corpus(embedder=None, target_size: int = 5000) -> list:
    """Static diverse corpus — no network, no generation needed."""
    seeds = [
        # Science & Technology
        "Quantum computers exploit superposition and entanglement to solve problems classical computers cannot.",
        "CRISPR-Cas9 allows precise editing of DNA sequences in living organisms.",
        "Machine learning models learn patterns from data without being explicitly programmed.",
        "The internet was originally designed as a decentralized communication network for resilience.",
        "Fusion energy promises nearly limitless clean power if containment can be sustained.",
        "Transistors shrank from centimeters to nanometers over seven decades of Moore's Law.",
        "Neural networks are loosely inspired by the structure of biological brains.",
        "Blockchain creates tamper-resistant records through distributed consensus mechanisms.",
        "Satellites in low Earth orbit now provide global broadband internet coverage.",
        "Autonomous vehicles must balance safety, speed, and ethical decision-making in real time.",
        "Large language models predict the next token based on context learned from vast text corpora.",
        "Robotics combines mechanical engineering, electronics, and software to automate physical tasks.",
        "Photovoltaic cells convert sunlight directly into electricity through the photoelectric effect.",
        "5G networks offer dramatically higher bandwidth and lower latency than previous generations.",
        "Cryptography protects digital communications using mathematical problems hard to reverse.",
        # Nature & Environment
        "Rainforests cover only six percent of Earth's surface but house over half of all species.",
        "Coral reefs support enormous biodiversity despite occupying a small fraction of the ocean floor.",
        "Climate change is shifting the geographic ranges of plant and animal species worldwide.",
        "Wolves reintroduced to Yellowstone triggered a cascade of ecological changes across the park.",
        "Bees pollinate roughly one third of the food crops humans depend on globally.",
        "Ocean acidification threatens shell-forming marine life as CO2 dissolves into seawater.",
        "Migratory birds navigate using magnetic fields, star patterns, and learned landmarks.",
        "Forests sequester carbon, regulate water cycles, and stabilize soil simultaneously.",
        "Deep-sea hydrothermal vents host ecosystems that thrive without sunlight.",
        "Wildfires, though destructive, are a natural part of many ecosystems' renewal cycles.",
        "The Amazon River discharges more freshwater into the ocean than any other river on Earth.",
        "Permafrost holds vast amounts of carbon that could accelerate warming if it thaws.",
        "Mangrove forests protect coastlines from storm surges and provide nurseries for fish.",
        "Soil microbiomes play crucial roles in nutrient cycling and plant health.",
        "The monarch butterfly migrates thousands of miles between Mexico and North America annually.",
        # History & Society
        "The printing press democratized knowledge by making books affordable and widely available.",
        "Industrialization transformed rural agrarian societies into urban manufacturing economies.",
        "The Cold War shaped geopolitics for decades through ideological competition and proxy conflicts.",
        "Ancient trade routes spread not just goods but also ideas, religions, and diseases.",
        "Democracy in Athens was limited to free male citizens, excluding women and slaves.",
        "The Scientific Revolution challenged centuries of religious authority over natural knowledge.",
        "Colonialism extracted wealth from subjugated peoples and reshaped global power dynamics.",
        "The French Revolution introduced concepts of liberty and popular sovereignty to Europe.",
        "World War II killed an estimated seventy to eighty-five million people worldwide.",
        "The civil rights movement used nonviolent protest to challenge institutionalized racial segregation.",
        "Urbanization has concentrated over half of humanity in cities for the first time in history.",
        "The Green Revolution dramatically increased agricultural yields in the twentieth century.",
        "Globalization has deepened economic interdependence while also creating new inequalities.",
        "Empires throughout history rose through military conquest and fell through overextension.",
        "Writing systems allowed complex societies to keep records, pass laws, and coordinate at scale.",
        # Health & Medicine
        "Vaccines work by training the immune system to recognize pathogens without causing disease.",
        "Antibiotics revolutionized medicine by making previously fatal bacterial infections treatable.",
        "The human gut microbiome influences immune function, mood, and metabolic health.",
        "Cancer arises when cells accumulate mutations that override normal growth controls.",
        "Mental health conditions are as real and debilitating as physical illnesses.",
        "Regular physical exercise reduces risk of cardiovascular disease, diabetes, and depression.",
        "Sleep deprivation impairs cognition, emotional regulation, and immune function.",
        "Nutrition shapes long-term health outcomes more than most other lifestyle factors.",
        "Gene therapy holds promise for treating inherited disorders by correcting faulty DNA.",
        "The placebo effect demonstrates how expectation and belief influence physical outcomes.",
        "Chronic stress elevates cortisol levels and accelerates cellular aging.",
        "Early diagnosis dramatically improves survival rates for most cancers.",
        "Telemedicine expanded access to healthcare for patients in remote and underserved areas.",
        "Organ transplantation requires lifelong immunosuppression to prevent rejection.",
        "Public health measures like clean water and sanitation saved more lives than most drugs.",
        # Philosophy & Ethics
        "Utilitarianism judges actions by the greatest happiness produced for the greatest number.",
        "Kant argued morality requires treating people as ends in themselves, never merely as means.",
        "Free will and determinism remain unresolved tensions in philosophy and neuroscience.",
        "Existentialism holds that individuals must create their own meaning in an indifferent universe.",
        "Justice requires both fairness in process and equity in outcomes.",
        "The trolley problem illustrates conflicts between consequentialist and deontological ethics.",
        "Plato believed knowledge of abstract Forms was more real than sensory experience.",
        "Stoicism teaches that virtue is the only true good and external events are beyond our control.",
        "Ethics of care prioritizes relationships and context over universal abstract principles.",
        "Moral relativism holds that ethical standards vary across cultures and periods.",
        "The nature of consciousness remains one of philosophy's most intractable problems.",
        "Political philosophy asks what justifies the authority of states over individuals.",
        "Rights-based ethics grounds morality in inalienable entitlements all humans possess.",
        "Nihilism denies any objective basis for meaning, value, or morality.",
        "Pragmatism evaluates ideas by their practical consequences rather than abstract truth.",
        # Economics & Business
        "Supply and demand determine prices in competitive markets through decentralized coordination.",
        "Inflation erodes purchasing power when the money supply grows faster than output.",
        "Startups disrupt incumbent industries by solving old problems with cheaper or better approaches.",
        "Network effects make platforms more valuable as more users join them.",
        "Behavioral economics shows humans systematically deviate from rational choice predictions.",
        "Compound interest rewards patient investors and punishes persistent borrowers.",
        "Trade deficits and surpluses reflect differences in savings rates and investment needs.",
        "Monopolies reduce consumer welfare by restricting output and raising prices.",
        "Automation displaces some jobs while creating demand for new kinds of work.",
        "Central banks use interest rates to balance inflation and unemployment objectives.",
        "Venture capital funds risky early-stage companies in exchange for equity stakes.",
        "Brand loyalty reduces price sensitivity and provides durable competitive advantages.",
        "Income inequality has risen in most developed economies since the 1980s.",
        "Microfinance extends small loans to entrepreneurs in developing countries.",
        "The gig economy offers flexibility but reduces worker protections and benefits.",
        # Psychology & Behavior
        "Cognitive biases cause systematic errors in judgment that are difficult to correct.",
        "Attachment styles formed in childhood influence relationship patterns throughout life.",
        "The bystander effect makes individuals less likely to help when others are present.",
        "Intrinsic motivation produces deeper engagement and creativity than external rewards.",
        "Habituation causes us to notice changes in our environment more than stable features.",
        "Confirmation bias leads people to seek information that supports existing beliefs.",
        "Emotions evolved as rapid signals to guide behavior in uncertain environments.",
        "Long-term memory is reconstructive, not reproductive — we rebuild rather than replay.",
        "Social identity shapes self-concept through group membership and comparison.",
        "Stress can enhance performance up to a threshold, beyond which it impairs it.",
        "Humans overestimate how much future events will affect their long-term happiness.",
        "Mirror neurons may underlie our ability to understand and imitate others' actions.",
        "Flow states arise when skill and challenge are in balance, producing deep absorption.",
        "Sleep consolidates memories by replaying neural patterns formed during waking hours.",
        "Growth mindset predicts academic achievement better than raw measured intelligence.",
        # Art, Culture & Language
        "Language shapes thought by structuring which distinctions are easy or hard to make.",
        "Music evokes emotion through expectation, tension, and resolution across cultures.",
        "Art serves as a mirror of society, reflecting its values, anxieties, and aspirations.",
        "Storytelling is one of humanity's oldest and most universal forms of communication.",
        "Architecture shapes behavior by determining how people move through and occupy space.",
        "Languages with more speakers tend to simplify their grammar over generations.",
        "Film editing creates meaning through juxtaposition that neither shot alone contains.",
        "Humor often works by violating expectations in a benign, non-threatening way.",
        "Cultural heritage preservation balances authenticity against accessibility and change.",
        "Translation is interpretation — no two languages map concepts onto the world identically.",
        "Improvisation in jazz requires internalizing rules deeply enough to break them creatively.",
        "Literary metaphors restructure how we understand abstract concepts.",
        "Dance is one of the few art forms that uses the human body as its primary medium.",
        "Typography influences how readers feel about content before they process its meaning.",
        "Oral traditions preserved complex knowledge across generations before writing existed.",
        # Space & Astronomy
        "The observable universe contains more stars than grains of sand on all Earth's beaches.",
        "Black holes warp spacetime so severely that not even light can escape their event horizon.",
        "The Big Bang did not occur at a point in space but was an expansion of space itself.",
        "Exoplanet detection has revealed that planetary systems are common throughout the galaxy.",
        "Dark matter outweighs ordinary matter five to one but has never been directly observed.",
        "Light from distant galaxies shows us the universe as it existed billions of years ago.",
        "Mars once had liquid water on its surface and may have harbored microbial life.",
        "Neutron stars pack more mass than the sun into a sphere the size of a city.",
        "Cosmic rays are high-energy particles that constantly bombard Earth from deep space.",
        "Gravitational waves ripple through spacetime when massive objects accelerate.",
        "The Voyager probes, launched in 1977, have now crossed into interstellar space.",
        "Jupiter's magnetic field is the largest structure in the solar system after the sun.",
        "Saturn's rings are primarily water ice and are remarkably thin relative to their width.",
        "The search for extraterrestrial intelligence scans radio frequencies for artificial signals.",
        "Stellar nucleosynthesis forged every element heavier than hydrogen and helium inside stars.",
        # Mathematics & Logic
        "Prime numbers have no factors other than one and themselves and are infinitely numerous.",
        "Gödel proved that any sufficiently powerful formal system contains true but unprovable statements.",
        "Probability theory quantifies uncertainty and underlies statistics, physics, and finance.",
        "Topology studies properties preserved under continuous deformation without tearing or gluing.",
        "The Pythagorean theorem relates the sides of right triangles in all Euclidean geometries.",
        "Chaos theory shows that tiny differences in initial conditions can lead to vastly different outcomes.",
        "Graph theory models networks of relationships and has applications across many fields.",
        "Infinity comes in different sizes — the real numbers are strictly more numerous than the integers.",
        "Bayesian reasoning updates probabilities as new evidence arrives.",
        "Linear algebra underlies computer graphics, machine learning, and quantum mechanics.",
        "Mathematical proof establishes certainty by deriving conclusions from axioms through logic.",
        "Fractals exhibit self-similarity at every scale, arising from simple iterative rules.",
        "Game theory analyzes strategic interactions where outcomes depend on multiple agents' choices.",
        "Complex numbers extend the real line into a plane and are essential in physics and engineering.",
        "The halting problem proves some questions are undecidable by any algorithm.",
        # Engineering & Design
        "Good design solves a problem simply while remaining intuitive for the intended user.",
        "Structural engineering must balance strength, weight, material cost, and aesthetic goals.",
        "Iterative prototyping reveals problems early when changes are still cheap to make.",
        "Redundancy in critical systems prevents single points of failure from causing catastrophe.",
        "User experience design centers on how people actually behave, not how designers expect them to.",
        "Civil engineering transformed cities by providing clean water, sewage, and reliable roads.",
        "The design of everyday objects encodes assumptions about who will use them and how.",
        "Feedback loops in control systems allow machines to self-correct toward desired states.",
        "Material science discovers new substances with properties tailored for specific applications.",
        "Sustainable engineering considers the full lifecycle environmental impact of designs.",
        "Elegant solutions in engineering often use fewer components to achieve the same function.",
        "Biomimicry applies principles from nature to solve human engineering challenges.",
        "Signal processing extracts useful information from noise in communications and sensors.",
        "Software architecture decisions made early constrain what is possible years later.",
        "Human factors engineering designs technology to match human cognitive and physical limits.",
    ]

    # Expand with paraphrases and variations by shuffling subsets
    expanded = list(seeds)
    # Add negations, questions, and partial variants for diversity
    extras = []
    for s in seeds[:100]:
        words = s.split()
        if len(words) > 8:
            extras.append(' '.join(words[:len(words)//2]) + '.')
    expanded.extend(extras)

    random.shuffle(expanded)
    # Deduplicate
    seen = set()
    corpus = []
    for s in expanded:
        s = s.strip()
        if s not in seen and len(s) >= 20:
            seen.add(s)
            corpus.append(s)

    random.shuffle(corpus)
    print(f"  Corpus: {len(corpus):,} sentences from static seed list")
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
        E = np.nan_to_num(E, nan=0.0, posinf=1.0, neginf=-1.0)
        chunk = 500
        sim   = np.zeros((n, n), dtype=np.float32)
        for i in range(0, n, chunk):
            ei = min(i + chunk, n)
            for j in range(0, n, chunk):
                ej = min(j + chunk, n)
                sim[i:ei, j:ej] = E[i:ei] @ E[j:ej].T
        sim = np.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=-1.0)
        sim = np.clip(sim, -1.0, 1.0)

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
            s      = float(sim[i, j])
            if np.isnan(s):
                attempts += 1
                continue
            bucket = min(int((s + 1.0) * 5), 9)  # map [-1,1] → [0,9]
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
    """Simple real-valued projection for stable pretraining.
    Learns to map Qwen embeddings to a 64-dim space that preserves similarity geometry.
    Weights are later transferred to the frame's DomainProjection."""
    def __init__(self, encoder=None, input_dim: int = 4096, d_model: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, d_model),
        )
        # Xavier init for stability
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_one(self, emb):
        x = self.proj(emb)
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def forward(self, a, b):
        ta  = self.encode_one(a)
        tb  = self.encode_one(b)
        sim = (ta * tb).sum(dim=-1)   # cosine sim of unit vectors
        return torch.clamp(sim, -1.0, 1.0)


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
            qs  = (embs[i] @ embs[j]).item()
            # tokens are unit-normalized by encode_one; dot = cosine similarity
            os_ = (tokens[i] @ tokens[j]).item()
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
    print(f"  Contrastive Pre-training v4 — BGE-small-en-v1.5 teacher")
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
                best_state = {k: v.clone() for k, v in student.state_dict().items()}
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

    TARGET_SIZE = 500   # use unique seeds only — no repetition needed
    N_PAIRS     = 5000
    BATCH_SIZE  = 256
    N_EPOCHS    = 60
    LR          = 2e-3
    INPUT_DIM   = 768    # BGE-small-en-v1.5 output dim

    # Load embedder
    embedder = BGEEmbedder(device)
    embedder.load()

    # Build / load corpus
    cache_path = 'corpus_embeddings_v4.pt'
    if os.path.exists(cache_path):
        print(f"\nLoading cached embeddings...")
        cache  = torch.load(cache_path, weights_only=True)
        corpus = cache['corpus']
        embs   = cache['embeddings']
        print(f"  {len(corpus):,} sentences, shape {tuple(embs.shape)}")
    else:
        print(f"\nBuilding corpus ({TARGET_SIZE:,} sentences)...")
        corpus = build_corpus(target_size=TARGET_SIZE)
        print(f"\nEncoding with BGE-small-en-v1.5...")
        t0   = time.perf_counter()
        embs = embedder.encode(corpus, batch_size=64)
        # Keep corpus in sync if any embeddings were dropped
        if len(embs) != len(corpus):
            corpus = corpus[:len(embs)]
        print(f"  Encoded {len(embs):,} sentences in {time.perf_counter()-t0:.1f}s")
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

    # Build student — simple real-valued projection, numerically stable
    student = StudentSimilarity(input_dim=INPUT_DIM, d_model=64)
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

    # Load best (if training improved at all)
    if best_state is not None:
        student.load_state_dict(best_state)

    # Final eval
    final_corr = evaluate(student, embedder, val_sentences, device)
    print(f"Baseline : {baseline:.4f}")
    print(f"Final    : {final_corr:.4f}  (+{final_corr - baseline:.4f})")

    # Save
    out_path = 'qwen_encoder_pretrained.pt'
    torch.save({
        'encoder_state': student.state_dict(),
        'd_model'      : 64,
        'k_sparse'     : 16,
        'input_dim'    : INPUT_DIM,
        'base_model'   : 'BAAI/bge-small-en-v1.5',
        'final_corr'   : final_corr,
        'corpus_size'  : len(corpus),
        'n_pairs'      : N_PAIRS,
    }, out_path)
    print(f"\nSaved: {out_path}")
