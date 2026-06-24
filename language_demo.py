"""
Omnicapable Transformer — Language Encoder Demo
================================================
Demonstrates how real text sentences flow through
the language domain encoder into the shared latent space.

Pipeline:
    Text sentence
        ↓
    sentence-transformers     ← pretrained, frozen, already understands language
    (all-MiniLM-L6-v2)
        ↓
    384-dim semantic vector   ← meaning encoded as geometry
        ↓
    Language DomainEncoder    ← our network maps to shared space
        ↓
    (batch, seq_len, 64)      ← complex sparse token, same format as price
    complex sparse token

Then shows:
    - Semantic similarity between sentences preserved in embedding space
    - How similar/opposite sentences produce geometrically related tokens
    - Price token + language token in the same space

Usage:
    python language_demo.py
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from token_encoder import DomainRegistry


# ─────────────────────────────────────────────────────────
#  SIMILARITY UTILITIES
# ─────────────────────────────────────────────────────────

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two real vectors."""
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()

def complex_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Similarity between two complex tokens.
    Uses magnitude-weighted cosine on the real part —
    dimensions with higher magnitude carry more weight.
    """
    # Use magnitude as the comparison signal
    mag_a = a.abs().flatten().float()
    mag_b = b.abs().flatten().float()
    sim = (mag_a @ mag_b / (mag_a.norm() * mag_b.norm() + 1e-8)).item()
    return sim


# ─────────────────────────────────────────────────────────
#  MAIN DEMO
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load pretrained sentence encoder ──
    print(f"\nLoading sentence-transformers model...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    st_model.to(device)
    print(f"Loaded: all-MiniLM-L6-v2  (384-dim embeddings)")

    # ── Build our domain registry ──
    registry = DomainRegistry(d_model=64, k_sparse=16)
    registry.register('language', input_dim=384)
    registry.register('price',    input_dim=12)
    registry.to(device)
    registry.eval()

    # ─────────────────────────────────────────────────────
    #  TEST 1: Financial sentences — semantic relationships
    #  We expect semantically similar sentences to produce
    #  geometrically similar tokens in our shared space.
    # ─────────────────────────────────────────────────────

    print(f"\n{'═'*62}")
    print(f"  TEST 1: Semantic similarity in shared space")
    print(f"{'═'*62}")

    sentence_groups = {
        'Hawkish / Rate hike': [
            "The Federal Reserve signals aggressive rate hikes ahead.",
            "Central bank raises interest rates to combat inflation.",
            "BOJ unexpectedly tightens monetary policy.",
        ],
        'Dovish / Rate cut': [
            "Federal Reserve hints at rate cuts amid economic slowdown.",
            "Central bank pivots to accommodative monetary policy.",
            "BOJ maintains ultra-loose policy stance.",
        ],
        'Risk-off / Fear': [
            "Markets plunge as recession fears grip investors.",
            "Panic selling accelerates across global equity markets.",
            "Safe haven assets surge amid geopolitical uncertainty.",
        ],
        'Risk-on / Bullish': [
            "Strong jobs report fuels optimism across risk assets.",
            "Equity markets rally on better than expected earnings.",
            "Investor sentiment reaches multi-year highs.",
        ],
    }

    # Encode all sentences
    all_sentences = []
    all_labels    = []
    for group_name, sentences in sentence_groups.items():
        for s in sentences:
            all_sentences.append(s)
            all_labels.append(group_name)

    print(f"\n  Encoding {len(all_sentences)} sentences through full pipeline...\n")

    # Get sentence-transformer embeddings (384-dim)
    with torch.no_grad():
        st_embeddings = st_model.encode(
            all_sentences,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False,
        )   # (N, 384)

        # Feed through our language encoder
        # Need shape (batch, seq_len, 384) — treat each sentence as seq_len=1
        st_embeddings_3d = st_embeddings.unsqueeze(1)   # (N, 1, 384)
        our_tokens = registry('language', st_embeddings_3d)  # (N, 1, 64) complex
        our_tokens_2d = our_tokens.squeeze(1)               # (N, 64) complex

    # Show sentence → token stats
    print(f"  {'Sentence':<52} {'|Mag|':>6}  {'Phase':>7}")
    print(f"  {'─'*66}")
    for i, (sent, label) in enumerate(zip(all_sentences, all_labels)):
        mag   = our_tokens_2d[i].abs().mean().item()
        phase = our_tokens_2d[i].angle().std().item()
        short = sent[:50] + '..' if len(sent) > 50 else sent
        print(f"  {short:<52} {mag:>6.3f}  {phase:>6.3f}r")

    # ── Intra-group vs inter-group similarity ──
    print(f"\n  Similarity matrix (our complex token space):")
    print(f"  Higher = more similar in our shared space\n")

    group_names  = list(sentence_groups.keys())
    group_tokens = {}
    idx = 0
    for gname, sentences in sentence_groups.items():
        n = len(sentences)
        group_tokens[gname] = our_tokens_2d[idx:idx+n]
        idx += n

    # Print pairwise group similarities
    print(f"  {'':28}", end='')
    for gname in group_names:
        short = gname[:12]
        print(f"  {short:>12}", end='')
    print()
    print(f"  {'─'*76}")

    for g1 in group_names:
        short1 = g1[:26]
        print(f"  {short1:<28}", end='')
        for g2 in group_names:
            # Mean token for each group
            mean1 = group_tokens[g1].mean(dim=0)
            mean2 = group_tokens[g2].mean(dim=0)
            sim   = complex_similarity(mean1, mean2)
            print(f"  {sim:>12.3f}", end='')
        print()

    print(f"\n  Diagonal = self-similarity (should be ~1.0)")
    print(f"  Hawkish vs Dovish should be lower than Hawkish vs Hawkish")

    # ─────────────────────────────────────────────────────
    #  TEST 2: Same space as price tokens
    #  Show that language and price tokens are compatible
    #  in the shared latent space.
    # ─────────────────────────────────────────────────────

    print(f"\n{'═'*62}")
    print(f"  TEST 2: Language + Price in the same space")
    print(f"{'═'*62}\n")

    # Fake price data — one window of 60 candles
    price_window = torch.randn(1, 60, 12).to(device)

    # One news headline relevant to that window
    headline = "BOJ signals sustained ultra-loose policy despite yen weakness."
    with torch.no_grad():
        # Encode headline
        hl_embed = st_model.encode(
            [headline],
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False,
        ).unsqueeze(1)   # (1, 1, 384)

        # Get tokens
        price_token = registry('price',    price_window)  # (1, 60, 64) complex
        lang_token  = registry('language', hl_embed)      # (1,  1, 64) complex

    print(f"  Headline : \"{headline}\"")
    print(f"\n  Price token  shape : {tuple(price_token.shape)}  dtype={price_token.dtype}")
    print(f"  Language token shape: {tuple(lang_token.shape)}  dtype={lang_token.dtype}")
    print(f"\n  ✓ Both are complex64")
    print(f"  ✓ Both have d_model=64")
    print(f"  ✓ Ready to concatenate into one sequence for the core loop")

    # Concatenate — price sequence + language token appended
    combined = torch.cat([price_token, lang_token], dim=1)  # (1, 61, 64)
    print(f"\n  Combined sequence shape: {tuple(combined.shape)}")
    print(f"  → 60 price tokens + 1 language token = 61 token sequence")
    print(f"  → Core loop attends across all 61 tokens simultaneously")

    # ─────────────────────────────────────────────────────
    #  TEST 3: Semantic opposites → different token geometry
    # ─────────────────────────────────────────────────────

    print(f"\n{'═'*62}")
    print(f"  TEST 3: Semantic opposites in token space")
    print(f"{'═'*62}\n")

    pairs = [
        ("Markets surge on strong economic data.",
         "Markets collapse amid recession fears."),
        ("BOJ raises rates aggressively.",
         "BOJ cuts rates to historic lows."),
        ("Investor confidence at all time high.",
         "Panic selling grips global markets."),
    ]

    for sent_a, sent_b in pairs:
        with torch.no_grad():
            emb = st_model.encode(
                [sent_a, sent_b],
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
            ).unsqueeze(1)  # (2, 1, 384)

            tokens = registry('language', emb).squeeze(1)  # (2, 64)

        # Similarity in ST space (384-dim real)
        st_sim  = cosine_similarity(emb[0].squeeze(), emb[1].squeeze())
        # Similarity in our space (64-dim complex)
        our_sim = complex_similarity(tokens[0], tokens[1])

        print(f"  A: \"{sent_a[:55]}\"")
        print(f"  B: \"{sent_b[:55]}\"")
        print(f"  ST space similarity  : {st_sim:>6.3f}")
        print(f"  Our space similarity : {our_sim:>6.3f}")
        print()

    print(f"{'═'*62}")
    print(f"  Language encoder ready.")
    print(f"  Next: combined price + language training task.")
    print(f"{'═'*62}\n")