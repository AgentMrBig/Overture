"""
Overture — Qwen3-VL-Embedding-2B Test
=======================================
Tests the Qwen3-VL-Embedding-2B model as our primary
language + vision domain part.

Key facts:
    - 2B parameters
    - 2048-dim output (can truncate to 64, 128, 256, 512, 1024)
    - Handles text, images, and mixed text+image inputs
    - Works via sentence-transformers
    - Completely free, runs locally on RTX

Tests:
    1. Text encoding — semantic quality vs BGE-large
    2. Image encoding — feed image URLs directly
    3. Cross-modal — text and image in same space
    4. Projection to our 64-dim complex space
    5. Truncation to 64-dim natively (skip our encoder entirely)

Usage:
    python qwen_vl_test.py
"""

import torch
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from token_encoder import DomainRegistry


def cosine_sim(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()

def complex_sim(a, b):
    mag_a = a.abs().flatten().float()
    mag_b = b.abs().flatten().float()
    return (mag_a @ mag_b / (mag_a.norm() * mag_b.norm() + 1e-8)).item()

def print_header(title):
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"{'═'*64}")

def print_section(title):
    print(f"\n  ── {title}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ─────────────────────────────────────────────────────
    #  LOAD QWEN3-VL-EMBEDDING-2B
    # ─────────────────────────────────────────────────────
    print(f"\nLoading Qwen3-VL-Embedding-2B...")
    print(f"(First run downloads ~4-5GB — subsequent runs load from cache)")
    t0 = time.perf_counter()

    try:
        model = SentenceTransformer(
            "Qwen/Qwen3-VL-Embedding-2B",
            trust_remote_code=True,
        )
        model.to(device)
        load_time = time.perf_counter() - t0
        print(f"Loaded in {load_time:.2f}s")
        EMBED_DIM = 2048
        print(f"Output dimension: {EMBED_DIM}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"\nTry updating sentence-transformers:")
        print(f"  pip install -U sentence-transformers")
        exit(1)

    # ─────────────────────────────────────────────────────
    #  TEST 1: TEXT ENCODING QUALITY
    #  Compare semantic discrimination against BGE-large
    # ─────────────────────────────────────────────────────
    print_header("TEST 1: Text Encoding Quality")

    text_pairs = [
        ("I am overflowing with joy and happiness.",
         "I am consumed by deep grief and sorrow.",
         "opposites"),
        ("The economy is growing strongly.",
         "The economy is collapsing into recession.",
         "opposites"),
        ("The dog barked at the stranger.",
         "The canine growled at the unknown visitor.",
         "similar"),
        ("Quantum mechanics describes subatomic particles.",
         "Particle physics studies fundamental matter.",
         "similar"),
        ("She laughed at the joke.",
         "The bridge spans the river.",
         "unrelated"),
        ("Music moves the human soul.",
         "Prime numbers have no factors.",
         "unrelated"),
    ]

    t0 = time.perf_counter()
    sentences_flat = [s for pair in text_pairs for s in pair[:2]]

    with torch.no_grad():
        embeddings = model.encode(
            sentences_flat,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
        embeddings = embeddings.to(device).float()  # cast bfloat16 → float32

    encode_time = time.perf_counter() - t0

    print(f"\n  Encoded {len(sentences_flat)} sentences in {encode_time:.3f}s")
    print(f"  Embedding shape: {tuple(embeddings.shape)}")
    print(f"\n  {'Pair':<45} {'Relation':<10} {'Similarity':>10}")
    print(f"  {'─'*68}")

    sims_by_type = {'opposites': [], 'similar': [], 'unrelated': []}
    for i, (a, b, rel) in enumerate(text_pairs):
        ea = embeddings[i*2]
        eb = embeddings[i*2+1]
        sim = cosine_sim(ea, eb)
        sims_by_type[rel].append(sim)
        label = f"{a[:22]}... / {b[:20]}..."
        print(f"  {label:<45} {rel:<10} {sim:>10.4f}")

    print(f"\n  Average similarities:")
    for rel, sims in sims_by_type.items():
        print(f"    {rel:<12}: {np.mean(sims):.4f}")

    gap = np.mean(sims_by_type['similar']) - np.mean(sims_by_type['opposites'])
    print(f"\n  Separation gap (similar - opposite): {gap:.4f}")
    print(f"  BGE-large achieved: 0.190  |  Target: >0.300")

    # ─────────────────────────────────────────────────────
    #  TEST 2: NATIVE 64-DIM TRUNCATION
    #  Qwen3-VL supports truncate_dim — output directly at
    #  64 dims without needing our encoder projection.
    #  Test if quality holds at 64 dims.
    # ─────────────────────────────────────────────────────
    print_header("TEST 2: Native 64-dim Truncation")
    print(f"  Testing quality at different output dimensions...")

    test_sents = [
        "Joy and happiness fill my heart.",
        "Deep sadness and grief overwhelm me.",
        "The dog ran quickly across the field.",
        "The canine sprinted rapidly through the meadow.",
        "Quantum mechanics is a branch of physics.",
        "She cooked pasta for dinner tonight.",
    ]

    dims_to_test = [2048, 512, 256, 128, 64]
    dim_results  = {}

    for dim in dims_to_test:
        t0 = time.perf_counter()
        try:
            model_trunc = SentenceTransformer(
                "Qwen/Qwen3-VL-Embedding-2B",
                trust_remote_code=True,
                truncate_dim=dim,
            )
            model_trunc.to(device)
            with torch.no_grad():
                embs = model_trunc.encode(
                    test_sents,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                if not isinstance(embs, torch.Tensor):
                    embs = torch.tensor(embs)
                embs = embs.to(device)

            # Joy vs Sadness (should be low)
            sim_opp = cosine_sim(embs[0], embs[1])
            # Dog sentences (should be high)
            sim_sim = cosine_sim(embs[2], embs[3])
            # Unrelated
            sim_unr = cosine_sim(embs[4], embs[5])

            elapsed = time.perf_counter() - t0
            dim_results[dim] = {
                'opposite': sim_opp,
                'similar': sim_sim,
                'unrelated': sim_unr,
                'gap': sim_sim - sim_opp,
                'time': elapsed,
            }
            print(f"  dim={dim:>5}  opp={sim_opp:.3f}  sim={sim_sim:.3f}  "
                  f"unrel={sim_unr:.3f}  gap={sim_sim-sim_opp:.3f}  "
                  f"({elapsed:.2f}s)")
        except Exception as e:
            print(f"  dim={dim:>5}  FAILED: {e}")

    # Find best dim
    best_dim = max(dim_results, key=lambda d: dim_results[d]['gap'])
    print(f"\n  Best gap at dim={best_dim}: {dim_results[best_dim]['gap']:.4f}")

    # ─────────────────────────────────────────────────────
    #  TEST 3: IMAGE ENCODING
    #  Feed image URLs directly — Qwen3-VL handles vision
    # ─────────────────────────────────────────────────────
    print_header("TEST 3: Image Encoding (Vision Domain)")
    print("  Generating local test images with PIL...")
    try:
        from PIL import Image
        import os
        img_warm = Image.new("RGB", (224, 224), color=(220, 80, 60))
        img_cool = Image.new("RGB", (224, 224), color=(60, 80, 220))
        img_warm.save("test_warm.jpg")
        img_cool.save("test_cool.jpg")
        print("  Created test_warm.jpg (red) and test_cool.jpg (blue)")
        t0 = time.perf_counter()
        inputs = [
            "test_warm.jpg",
            "test_cool.jpg",
            "A warm red sunset glowing over the horizon.",
            "A cool blue ocean stretching to the horizon.",
        ]
        img_labels = ["Warm image", "Cool image", "Warm text", "Cool text"]
        with torch.no_grad():
            img_embeddings = model.encode(
                inputs,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if not isinstance(img_embeddings, torch.Tensor):
                img_embeddings = torch.tensor(img_embeddings)
            img_embeddings = img_embeddings.to(device).float()
        elapsed = time.perf_counter() - t0
        print(f"  Encoded 2 images + 2 text in {elapsed:.2f}s")
        print(f"  Shape: {tuple(img_embeddings.shape)}")
        warm_img  = img_embeddings[0]
        cool_img  = img_embeddings[1]
        warm_text = img_embeddings[2]
        cool_text = img_embeddings[3]
        emb_list = [warm_img, cool_img, warm_text, cool_text]
        print("  Cross-modal similarity matrix:")
        for lbl, ei in zip(img_labels, emb_list):
            row = f"    {lbl:<14}"
            for ej in emb_list:
                row += f"  {cosine_sim(ei, ej):.4f}"
            print(row)
        warm_cross = cosine_sim(warm_img, warm_text)
        cool_cross = cosine_sim(warm_img, cool_text)
        print(f"  Warm image to Warm text : {warm_cross:.4f}")
        print(f"  Warm image to Cool text : {cool_cross:.4f}")
        result = "Working" if warm_cross > cool_cross else "Not aligned"
        print(f"  Cross-modal alignment   : {result}")
        os.remove("test_warm.jpg")
        os.remove("test_cool.jpg")
    except Exception as e:
        print(f"  Image test failed: {e}")
        import traceback
        traceback.print_exc()
