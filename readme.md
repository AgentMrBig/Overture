# Project Overture

> *An omnicapable Transformer built from scratch — complex-valued, sparse, iterative, and domain-agnostic.*

---

## What Is This

Project Overture is a custom Transformer architecture designed from first principles. It is not a wrapper around an existing model. It is not a fine-tuning framework. Every weight, every forward pass, every design decision is built from scratch in PyTorch.

The core thesis: **a small network that thinks deeply beats a large network that thinks once.**

Instead of achieving capability through billions of parameters, Overture achieves capability through:

- **Complex-valued tokens** — magnitude encodes presence, phase encodes relational position
- **Learned sparsity** — only the most relevant dimensions activate per token
- **Weight-tied iterative loops** — the same block runs N times, depth comes from iteration not parameters
- **Domain-agnostic shared space** — text, images, audio, price data all map to the same token format
- **Pretrained domain parts** — frozen specialist models bolt onto the frame, no retraining needed

---

## Architecture

```
ANY INPUT (text, image, audio, price, ...)
        |
Pretrained Domain Part      -- frozen specialist (Qwen3-VL, Whisper, etc.)
        |
Domain Encoder              -- lightweight learned projection
                               maps embedding -> complex sparse token
        |
Complex Sparse Token        -- shape: (batch, seq_len, d_model) complex64
                               magnitude: feature presence
                               phase:     relational/temporal position
                               sparse:    top-k dims active per token
        |
+--- CORE LOOP (weight-tied) -------------------------------------------+
|                                                                        |
|   Complex Multi-Head Attention                                         |
|   scores = Re(Q . K_conj_T) / sqrt(d)  <- conjugate inner product    |
|        |                                                               |
|   Complex Feed-Forward   (ModReLU -- preserves phase)                 |
|        |                                                               |
|   Convergence Check  <- exit early if representation stabilized       |
|   loop again or exit                                                   |
+------------------------------------------------------------------------+
        |
Output Head                 -- pool -> complex->real -> projection
                               binary / multiclass / regression / sequence
        |
PREDICTION
```

---

## The Machine Parts Philosophy

The frame does not need to learn language, vision, or audio from scratch. Pretrained specialist models are bolted on as frozen domain parts -- like high-precision machine components attached to a custom frame. The frame learns what to *do* with their outputs, not how to produce them.

```
Qwen3-VL-Embedding-2B    -> text and image understanding (2048-dim, Apache 2.0)
Whisper encoder           -> audio and speech understanding (planned)
Custom price encoder      -> OHLCV financial time series (trained from scratch)
```

Each part outputs a vector. The domain encoder maps it into the shared complex space. The core loop attends across all domains simultaneously.

---

## Key Design Decisions

**Why complex-valued tokens?**
A complex number has magnitude and phase. Magnitude encodes how strongly a feature is present. Phase encodes relational and temporal position. For any signal data -- price, audio, language -- phase relationships carry information that real-valued vectors discard entirely.

**Why sparse?**
Different inputs should activate different subspaces. Learned top-k sparsity forces specialization to emerge from training. It also maps perfectly to Blackwell hardware-accelerated structured sparsity.

**Why weight-tied loops?**
A 96-layer Transformer has 96 sets of unique weights. A looped Transformer with 1 block run 96 times has 1 set. The looped model learns weights useful at every stage of refinement. Adaptive convergence gating means easy inputs exit in 2-3 loops, hard inputs run longer.

**Why domain-agnostic?**
The goal is not a trading model or a language model. It is an architecture that handles any problem domain by registering a new encoder. The core loop never changes. New domains plug in.

---

## Files

```
Overture/
|-- token_encoder.py           # Domain encoders -> complex sparse tokens
|-- core_loop.py               # Weight-tied iterative Transformer
|-- output_head.py             # Task-specific output heads
|-- training_loop.py           # Full training pipeline
|-- language_demo.py           # BGE-large language encoder demo
|-- bge_test.py                # BGE-large vs MiniLM quality comparison
|-- contrastive_pretrain.py    # Language encoder pre-trainer v1
|-- contrastive_pretrain_v2.py # Language encoder pre-trainer v2 (40k sentences)
`-- qwen_vl_test.py            # Qwen3-VL-Embedding-2B full test suite
```

---

## Quickstart

**1. Create and activate virtual environment**
```bash
python -m venv overture_env
source overture_env/Scripts/activate   # Windows Git Bash
source overture_env/bin/activate       # Mac / Linux
```

**2. Install dependencies**
```bash
# PyTorch nightly cu128 -- required for RTX 5060 (Blackwell sm_120)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# For older GPUs use stable with your CUDA version
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install sentence-transformers datasets numpy pillow
```

**3. Run in order**
```bash
python token_encoder.py
python core_loop.py
python output_head.py
python training_loop.py
python qwen_vl_test.py
```

---

## Results

### Synthetic Trend Classification

| Metric | Value |
|---|---|
| Parameters | 123,138 |
| Epochs to convergence | 5 |
| Final val accuracy | 100.0% |
| Prediction confidence | 99.8 - 99.9% |
| Hardware | NVIDIA RTX 5060 (Blackwell) |

### Qwen3-VL-Embedding-2B Integration

| Test | Result |
|---|---|
| Text encoding speed | 12 sentences in 0.43s |
| Separation gap at 2048-dim | 0.311 (vs BGE-large 0.190) |
| Separation gap at 64-dim | 0.303 -- beats BGE-large with 32x compression |
| Cross-modal image + text alignment | Working |
| Frame integration | (batch, seq, 64) complex64 confirmed |

**Key finding:** Qwen3-VL with truncate_dim=64 achieves a better separation gap than BGE-large at full 1024-dims. The domain projection layer can be bypassed entirely for this model.

---

## Research Directions

**Complex Ternary Attention**
Ternary weights ({-1, 0, +1}) applied to complex-valued weights produce 9 discrete values encoding rotations in the complex plane. No multiplication needed -- only addition and subtraction. Hypothesis: complex ternary preserves semantic structure better than real ternary on temporal data. Unpublished combination.

**Phase Coherence Attention**
Tokens with aligned phase interfere constructively. Opposite phase cancels. For temporal data this captures structural information dot-product attention misses. A qualitative difference, not just a performance tweak.

**FFT Attention on Native Complex Sequences**
O(n log n) vs O(n^2). Our tokens are already complex -- no conversion needed. Novel application to natively complex token sequences.

**Adaptive Compute via Convergence Gating**
Loop count scales with problem difficulty. Average inference cost drops ~60% with no accuracy loss.

**Hardware-Aligned Architecture**
Every dimension is a multiple of 16 (d_model=64, k_sparse=16, n_heads=4, ff_dim=256). Aligns perfectly with Blackwell tensor cores and structured sparsity acceleration.

**Self-Modifying Networks (Level 5)**
Long-term research target. The looping architecture is a natural substrate for self-reasoning. An evaluation gate enforces that only genuinely better versions are kept. Build order: canvas environment (Level 4) -> recursive self-improvement with gate (Level 5).

---

## Roadmap

- [x] Complex sparse token encoder
- [x] Weight-tied iterative core loop with convergence detection
- [x] Multi-task output head (binary, multiclass, regression, sequence)
- [x] Training pipeline -- 100% accuracy in 5 epochs on synthetic task
- [x] Language encoder via BGE-large + contrastive pre-training
- [x] Qwen3-VL-Embedding-2B as primary language + vision part
- [x] Cross-modal image + text alignment confirmed
- [x] 64-dim truncation beats BGE-large at full 1024-dim
- [ ] Architectural bypass mode for direct-dimension models
- [ ] REPL terminal -- interactive frame interface
- [ ] Contrastive pre-training to >0.85 correlation
- [ ] Real OHLCV price data pipeline
- [ ] Whisper audio encoder integration
- [ ] Combined multimodal training task
- [ ] Complex ternary core loop (primary research contribution)
- [ ] LoRA adapter infrastructure
- [ ] FFT attention on complex sequences
- [ ] Phase coherence attention
- [ ] Canvas environment + code generation (Level 4 self-modification)
- [ ] Recursive self-improvement with evaluation gate (Level 5)

---

## Hardware Notes

Developed on:
- **GPU**: NVIDIA GeForce RTX 5060 (Blackwell sm_120, 8GB VRAM)
- **RAM**: 50GB
- **CUDA**: 13.1
- **PyTorch**: nightly cu128
- **Python**: 3.10.6
- **OS**: Windows 11, Git Bash

RTX 50-series requires PyTorch nightly cu128. Standard stable releases do not support sm_120. For RTX 30xx/40xx use stable cu118/cu121/cu124.

---

## Dependencies

| Library | Purpose |
|---|---|
| torch (nightly cu128) | Core engine |
| sentence-transformers | Qwen3-VL, BGE-large |
| datasets | HuggingFace streaming |
| numpy | Data processing |
| pillow | Image generation for vision tests |

Coming soon: openai-whisper, pandas, scikit-learn, plotly

---

*Project Overture -- built from scratch, one layer at a time.*