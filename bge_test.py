"""
Overture — BGE-large Quality Test
===================================
Tests BGE-large (1024-dim) against MiniLM (384-dim) on:
    1. Semantic similarity — do opposites separate better?
    2. Analogy structure  — does king - man + woman ≈ queen?
    3. Concept clustering — do mixed concepts group correctly?
    4. Survival test      — how much structure survives our
                           projection into 64-dim complex space?

Usage:
    python bge_test.py
"""

import torch
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from token_encoder import DomainRegistry

_test_times = {}
_t_start_global = None

def tick(label):
    _test_times[label] = time.perf_counter()

def tock(label):
    elapsed = time.perf_counter() - _test_times[label]
    print(f"\n  ⏱  {label}: {elapsed:.3f}s")
    return elapsed


# ─────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()

def complex_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    mag_a = a.abs().flatten().float()
    mag_b = b.abs().flatten().float()
    return (mag_a @ mag_b / (mag_a.norm() * mag_b.norm() + 1e-8)).item()

def encode_sentences(model, sentences, device):
    """Encode sentences with a ST model → (N, dim) tensor."""
    with torch.no_grad():
        return model.encode(
            sentences,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

def to_complex_tokens(registry, domain, embeddings):
    """Push ST embeddings through our domain encoder → complex tokens."""
    with torch.no_grad():
        emb_3d = embeddings.unsqueeze(1)          # (N, 1, dim)
        tokens = registry(domain, emb_3d)          # (N, 1, 64) complex
        return tokens.squeeze(1)                   # (N, 64) complex

def print_header(title):
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"{'═'*64}")

def print_section(title):
    print(f"\n  ── {title} {'─'*(54-len(title))}")


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    t_global = time.perf_counter()

    # ── Load both models ──
    print(f"\nLoading models...")
    tick('model_load')
    mini  = SentenceTransformer('all-MiniLM-L6-v2')
    bge   = SentenceTransformer('BAAI/bge-large-en-v1.5')
    mini.to(device)
    bge.to(device)
    tock('model_load')
    print(f"  MiniLM  : all-MiniLM-L6-v2   (384-dim)")
    print(f"  BGE     : BAAI/bge-large-en-v1.5 (1024-dim)")

    # ── Build two registries — one per model ──
    reg_mini = DomainRegistry(d_model=64, k_sparse=16)
    reg_mini.register('lang', input_dim=384)
    reg_mini.to(device).eval()

    reg_bge = DomainRegistry(d_model=64, k_sparse=16)
    reg_bge.register('lang', input_dim=1024)
    reg_bge.to(device).eval()

    # ═════════════════════════════════════════════════════
    #  TEST 1: SEMANTIC SIMILARITY — OPPOSITES
    #  We want opposite pairs to score LOW and
    #  similar pairs to score HIGH.
    #  BGE-large should show wider separation.
    # ═════════════════════════════════════════════════════

    print_header("TEST 1: Semantic Similarity — Opposite Pairs")
    tick('test1')

    pairs = [
        # (sentence_a, sentence_b, expected_relation)
        ("The sun is shining brightly.",
         "It is a dark stormy night.",
         "opposites"),
        ("I am filled with joy and happiness.",
         "I am consumed by grief and sorrow.",
         "opposites"),
        ("The economy is booming with strong growth.",
         "The economy is collapsing into deep recession.",
         "opposites"),
        ("Trust is the foundation of all relationships.",
         "Betrayal destroys the bonds between people.",
         "opposites"),
        ("A dog is a loyal domestic animal.",
         "A wolf is a wild predatory animal.",
         "similar"),
        ("The ocean is vast and deep.",
         "The sea stretches endlessly before us.",
         "similar"),
        ("Quantum mechanics describes subatomic particles.",
         "Particle physics studies fundamental matter.",
         "similar"),
    ]

    print(f"\n  {'Pair':<48} {'Rel':<10} {'MiniLM':>8} {'BGE':>8} {'Winner':>8}")
    print(f"  {'─'*84}")

    mini_scores, bge_scores = [], []

    for sent_a, sent_b, relation in pairs:
        emb_mini = encode_sentences(mini, [sent_a, sent_b], device)
        emb_bge  = encode_sentences(bge,  [sent_a, sent_b], device)

        tok_mini = to_complex_tokens(reg_mini, 'lang', emb_mini)
        tok_bge  = to_complex_tokens(reg_bge,  'lang', emb_bge)

        sim_mini = complex_sim(tok_mini[0], tok_mini[1])
        sim_bge  = complex_sim(tok_bge[0],  tok_bge[1])

        mini_scores.append((sim_mini, relation))
        bge_scores.append((sim_bge, relation))

        # For opposites, lower is better. For similar, higher is better.
        if relation == 'opposites':
            winner = 'BGE' if sim_bge < sim_mini else 'MiniLM'
        else:
            winner = 'BGE' if sim_bge > sim_mini else 'MiniLM'

        label = f"{sent_a[:22]}... / {sent_b[:20]}..."
        print(f"  {label:<48} {relation:<10} {sim_mini:>8.3f} {sim_bge:>8.3f} {winner:>8}")

    # Summary
    opp_mini = np.mean([s for s, r in mini_scores if r == 'opposites'])
    opp_bge  = np.mean([s for s, r in bge_scores  if r == 'opposites'])
    sim_mini_avg = np.mean([s for s, r in mini_scores if r == 'similar'])
    sim_bge_avg  = np.mean([s for s, r in bge_scores  if r == 'similar'])

    print(f"\n  Average opposite similarity  — MiniLM: {opp_mini:.3f}  BGE: {opp_bge:.3f}")
    print(f"  Average similar  similarity  — MiniLM: {sim_mini_avg:.3f}  BGE: {sim_bge_avg:.3f}")
    print(f"  Separation gap (sim - opp)   — MiniLM: {sim_mini_avg-opp_mini:.3f}  BGE: {sim_bge_avg-opp_bge:.3f}")
    print(f"  (Larger gap = better discrimination)")
    tock('test1')

    # ═════════════════════════════════════════════════════
    #  TEST 2: ANALOGY STRUCTURE
    #  Classic word vector analogy test.
    #  king - man + woman ≈ queen
    #  We test this in our complex token space.
    #  If the geometry is preserved, analogy arithmetic works.
    # ═════════════════════════════════════════════════════

    print_header("TEST 2: Analogy Structure in Complex Space")
    tick('test2')
    print(f"  Testing: A - B + C ≈ D  (does the geometry hold?)\n")

    analogies = [
        ("king",    "man",      "woman",   "queen"),
        ("paris",   "france",   "germany", "berlin"),
        ("hot",     "fire",     "ice",     "cold"),
        ("doctor",  "hospital", "school",  "teacher"),
        ("fast",    "cheetah",  "turtle",  "slow"),
        ("night",   "dark",     "bright",  "day"),
    ]

    print(f"  {'Analogy':<36} {'MiniLM rank':>12} {'BGE rank':>10}")
    print(f"  {'─'*60}")

    # For each analogy, encode a candidate pool and find nearest neighbor
    candidate_pool = list(set(
        word for analogy in analogies for word in analogy
    ))

    emb_pool_mini = encode_sentences(mini, candidate_pool, device)
    emb_pool_bge  = encode_sentences(bge,  candidate_pool, device)
    tok_pool_mini = to_complex_tokens(reg_mini, 'lang', emb_pool_mini)
    tok_pool_bge  = to_complex_tokens(reg_bge,  'lang', emb_pool_bge)
    pool_idx      = {w: i for i, w in enumerate(candidate_pool)}

    for A, B, C, D in analogies:
        # Compute A - B + C in complex token space
        def analogy_vector(pool_tokens, pool_index, a, b, c):
            va = pool_tokens[pool_index[a]]
            vb = pool_tokens[pool_index[b]]
            vc = pool_tokens[pool_index[c]]
            return va - vb + vc

        target_mini = analogy_vector(tok_pool_mini, pool_idx, A, B, C)
        target_bge  = analogy_vector(tok_pool_bge,  pool_idx, A, B, C)

        # Find nearest neighbor (excluding A, B, C themselves)
        exclude = {A, B, C}

        def find_rank(pool_tokens, pool_index, target, answer, exclude_set):
            sims = []
            for word, idx in pool_index.items():
                if word in exclude_set:
                    continue
                s = complex_sim(target, pool_tokens[idx])
                sims.append((s, word))
            sims.sort(reverse=True)
            words_ranked = [w for _, w in sims]
            rank = words_ranked.index(answer) + 1 if answer in words_ranked else -1
            top1 = words_ranked[0] if words_ranked else '?'
            return rank, top1

        rank_mini, top_mini = find_rank(tok_pool_mini, pool_idx, target_mini, D, exclude)
        rank_bge,  top_bge  = find_rank(tok_pool_bge,  pool_idx, target_bge,  D, exclude)

        correct_mini = '✓' if rank_mini == 1 else f'#{rank_mini}({top_mini})'
        correct_bge  = '✓' if rank_bge  == 1 else f'#{rank_bge}({top_bge})'

        label = f"{A} - {B} + {C} = {D}"
        print(f"  {label:<36} {correct_mini:>12} {correct_bge:>10}")
    tock('test2')

    # ═════════════════════════════════════════════════════
    #  TEST 3: CONCEPT CLUSTERING
    #  Throw 24 mixed concepts at both models.
    #  Do they naturally cluster by category
    #  in our complex token space?
    # ═════════════════════════════════════════════════════

    print_header("TEST 3: Concept Clustering in Complex Space")
    tick('test3')

    concepts = {
        'Animals'   : ["lion", "elephant", "dolphin", "eagle", "python", "wolf"],
        'Emotions'  : ["joy", "grief", "anger", "serenity", "fear", "love"],
        'Science'   : ["quantum", "entropy", "gravity", "evolution", "relativity", "photon"],
        'Actions'   : ["running", "sleeping", "building", "destroying", "creating", "learning"],
    }

    all_concepts  = []
    all_labels    = []
    for category, words in concepts.items():
        for word in words:
            all_concepts.append(word)
            all_labels.append(category)

    emb_mini_c = encode_sentences(mini, all_concepts, device)
    emb_bge_c  = encode_sentences(bge,  all_concepts, device)
    tok_mini_c = to_complex_tokens(reg_mini, 'lang', emb_mini_c)
    tok_bge_c  = to_complex_tokens(reg_bge,  'lang', emb_bge_c)

    def cluster_purity(tokens, labels):
        """
        For each concept, find its nearest neighbor.
        Purity = fraction where nearest neighbor shares the same category.
        """
        correct = 0
        n = len(tokens)
        for i in range(n):
            best_sim = -1
            best_j   = -1
            for j in range(n):
                if i == j:
                    continue
                s = complex_sim(tokens[i], tokens[j])
                if s > best_sim:
                    best_sim = s
                    best_j   = j
            if labels[best_j] == labels[i]:
                correct += 1
        return correct / n

    purity_mini = cluster_purity(tok_mini_c, all_labels)
    purity_bge  = cluster_purity(tok_bge_c,  all_labels)

    print(f"\n  Nearest-neighbor purity (same category = correct):")
    print(f"  MiniLM : {purity_mini:.1%}")
    print(f"  BGE    : {purity_bge:.1%}")
    print(f"  (100% = every concept's nearest neighbor is in its category)")

    # Show intra vs inter category similarity
    print(f"\n  Intra-category vs inter-category similarity (BGE):\n")
    categories = list(concepts.keys())
    cat_tokens = {}
    idx = 0
    for cat, words in concepts.items():
        n = len(words)
        cat_tokens[cat] = tok_bge_c[idx:idx+n]
        idx += n

    print(f"  {'':16}", end='')
    for c in categories:
        print(f"  {c[:10]:>10}", end='')
    print()
    print(f"  {'─'*56}")

    for c1 in categories:
        print(f"  {c1:<16}", end='')
        for c2 in categories:
            sims = []
            for i in range(len(cat_tokens[c1])):
                for j in range(len(cat_tokens[c2])):
                    if c1 == c2 and i == j:
                        continue
                    sims.append(complex_sim(cat_tokens[c1][i], cat_tokens[c2][j]))
            avg = np.mean(sims)
            marker = '●' if c1 == c2 else ' '
            print(f"  {avg:>9.3f}{marker}", end='')
        print()

    print(f"\n  ● = within-category similarity (should be highest in each row)")
    tock('test3')

    # ═════════════════════════════════════════════════════
    #  TEST 4: STRUCTURE SURVIVAL
    #  How much semantic structure survives the projection
    #  from high-dim ST space → 64-dim complex space?
    #  Compare ST-space similarity vs our-space similarity
    #  across many pairs. High correlation = good survival.
    # ═════════════════════════════════════════════════════

    print_header("TEST 4: Structure Survival Through Projection")
    tick('test4')
    print(f"  Correlation between ST-space and our complex space similarity")
    print(f"  High correlation = semantic structure survives compression\n")

    test_sentences = [
        "The stars shine at night.",
        "Darkness falls when the sun sets.",
        "Mathematics is the language of the universe.",
        "Numbers describe the fabric of reality.",
        "A child laughs with pure joy.",
        "Grief is the price of love.",
        "The ocean is ancient and unknowable.",
        "Mountains stand silent through the ages.",
        "Courage means acting despite fear.",
        "Wisdom comes from lived experience.",
        "Music transcends language and culture.",
        "Art expresses what words cannot.",
    ]

    emb_mini_s = encode_sentences(mini, test_sentences, device)
    emb_bge_s  = encode_sentences(bge,  test_sentences, device)
    tok_mini_s = to_complex_tokens(reg_mini, 'lang', emb_mini_s)
    tok_bge_s  = to_complex_tokens(reg_bge,  'lang', emb_bge_s)

    n = len(test_sentences)
    st_sims_mini, our_sims_mini = [], []
    st_sims_bge,  our_sims_bge  = [], []

    for i in range(n):
        for j in range(i+1, n):
            st_sims_mini.append(cosine_sim(emb_mini_s[i], emb_mini_s[j]))
            our_sims_mini.append(complex_sim(tok_mini_s[i], tok_mini_s[j]))
            st_sims_bge.append(cosine_sim(emb_bge_s[i], emb_bge_s[j]))
            our_sims_bge.append(complex_sim(tok_bge_s[i], tok_bge_s[j]))

    corr_mini = np.corrcoef(st_sims_mini, our_sims_mini)[0, 1]
    corr_bge  = np.corrcoef(st_sims_bge,  our_sims_bge)[0, 1]

    print(f"  MiniLM → our space correlation : {corr_mini:.4f}")
    print(f"  BGE    → our space correlation : {corr_bge:.4f}")
    print(f"\n  Interpretation:")
    print(f"  > 0.90  Excellent — almost all structure preserved")
    print(f"  > 0.75  Good      — most structure preserved")
    print(f"  > 0.50  Fair      — significant compression loss")
    print(f"  < 0.50  Poor      — structure not surviving projection")
    tock('test4')

    # ═════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═════════════════════════════════════════════════════

    print_header("SUMMARY")
    print(f"\n  MiniLM (384-dim)  →  64-dim complex space:")
    print(f"    Opposite separation gap : {sim_mini_avg - opp_mini:.3f}")
    print(f"    Cluster purity          : {purity_mini:.1%}")
    print(f"    Structure survival corr : {corr_mini:.4f}")
    print(f"\n  BGE-large (1024-dim) →  64-dim complex space:")
    print(f"    Opposite separation gap : {sim_bge_avg - opp_bge:.3f}")
    print(f"    Cluster purity          : {purity_bge:.1%}")
    print(f"    Structure survival corr : {corr_bge:.4f}")

    better = "BGE-large" if corr_bge > corr_mini else "MiniLM"
    print(f"\n  Better model for our frame : {better}")
    total = time.perf_counter() - t_global
    print(f"\n  Total runtime : {total:.2f}s")
    t1 = _test_times.get('test1', 0)
    print(f"  ├── Model load : {_test_times.get('model_load',0) - t_global + (time.perf_counter() - _test_times.get('test4',time.perf_counter())):.2f}s  (cached = near zero)")
    print(f"{'═'*64}")
    print(f"  Tests complete. Ready to build the REPL.")
    print(f"{'═'*64}\n")