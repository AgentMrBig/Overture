# Project Overture

> *An omnicapable Transformer built from scratch — complex-valued, sparse, iterative, and domain-agnostic.*

---

## What Is This

Project Overture is a custom Transformer architecture designed from first principles. It is not a wrapper around an existing model. It is not a fine-tuning framework. Every weight, every forward pass, every design decision is built from scratch in PyTorch.

The core thesis: **a small network that thinks deeply beats a large network that thinks once.**

Instead of achieving capability through billions of parameters, Overture achieves capability through:

- **Complex-valued tokens** -- magnitude encodes presence, phase encodes relational position
- **Learned sparsity** -- only the most relevant dimensions activate per token
- **Weight-tied iterative loops** -- the same block runs N times, depth comes from iteration not parameters
- **Domain-agnostic shared space** -- text, images, audio, price data all map to the same token format
- **Pretrained domain parts** -- frozen specialist models bolt onto the frame like machine parts, no retraining needed

---

## Architecture

```
ANY INPUT (text, image, audio, price, ...)
        |
Pretrained Domain Part      -- frozen specialist (Qwen3-VL, Whisper, etc.)
        |                      converts raw input -> high-dim embedding
Domain Encoder              -- lightweight learned projection
        |                      maps embedding -> complex sparse token
Complex Sparse Token        -- shape: (batch, seq_len, d_model) complex64
        |                      magnitude: feature presence
        |                      phase:     relational/temporal position
        |                      sparse:    top-k dims active per token
+--- CORE LOOP (weight-tied, runs N times) ---+
|   Complex Multi-Head Attention              |
|   Complex Feed-Forward (ModReLU)            |
|   Convergence Check -> exit or loop again  |
+---------------------------------------------+
        |
Output Head                 -- pool -> complex->real -> task prediction
        |
PREDICTION
```

---

## The Machine Parts Philosophy

The frame does not need to learn language, vision, or audio from scratch. Pretrained specialist models are bolted on as frozen domain parts. The frame learns what to *do* with their outputs, not how to produce them.

```
Qwen3-VL-Embedding-2B    -> text and image understanding (64-dim truncated, Apache 2.0)
Whisper encoder           -> audio and speech understanding (planned)
Custom price encoder      -> OHLCV financial time series (trained from scratch)
```

Each part outputs a vector. The domain encoder lifts it into complex space. The core loop attends across all domains simultaneously.

---

## How the REPL Results Work

When you type a command like `similarity "joy" "grief"`, here is the exact chain:

**Step 1 -- Qwen3-VL encodes the text**
The 2 billion parameter frozen model reads both inputs and produces 64-dimensional real vectors (truncated from 2048). These vectors carry deep semantic understanding learned from hundreds of millions of text examples. Joy and grief being emotionally related, blankets being warm and comforting, king and queen sharing royalty -- all of that is already in the geometry.

**Step 2 -- Domain encoder lifts to complex space**
Our 16k parameter domain encoder projects the real vectors through ComplexLift (polar mode), converting to magnitude + phase representation. The SparsityMask selects the top 16 most active dimensions, producing 64-dim complex sparse tokens.

**Step 3 -- Core loop refines**
The weight-tied attention block runs up to 8 iterations. Each loop the representation refines -- deltas dropping from ~0.6 toward 0.0 as the network finds a stable encoding. You can watch this in real time with the `loops` command.

**Step 4 -- Similarity computed**
Magnitude-weighted cosine similarity between complex tokens. Measures how much the same dimensions are active at similar magnitudes in both tokens.

**The key insight:** semantic structure in the REPL results comes entirely from frozen Qwen3-VL weights. Our encoder and core loop are currently randomly initialized. After contrastive pre-training brings structure survival above 0.85, results will be significantly sharper.

---

## REPL Commands

```
encode "text"                  encode a sentence -> token stats, loop info
encode path/to/image.jpg       encode an image   -> same stats
similarity "text a" "text b"   similarity in complex space
compare "a" "b" "c" "d"        pairwise similarity matrix
cluster word1 word2 word3 ...  group concepts by nearest neighbor
search "query" "c1" "c2" ...   rank candidates by similarity to query
loops "text"                   watch loop-by-loop convergence in real time
status                         model config, VRAM, parameter count
help                           command list
exit                           quit
```

---

## Files

```
Overture/
|-- token_encoder.py           # Domain encoders -> complex sparse tokens
|-- core_loop.py               # Weight-tied iterative Transformer
|-- output_head.py             # Task-specific output heads
|-- training_loop.py           # Full training pipeline
|-- language_demo.py           # BGE-large language encoder demo
|-- bge_test.py                # BGE-large vs MiniLM quality tests
|-- contrastive_pretrain.py    # Language encoder pre-trainer v1
|-- contrastive_pretrain_v2.py # Language encoder pre-trainer v2 (40k sentences)
|-- qwen_vl_test.py            # Qwen3-VL full test suite
`-- overture.py                # REPL -- main entry point
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

**3. Launch the REPL**
```bash
python overture.py
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

### Qwen3-VL Integration

| Test | Result |
|---|---|
| Text encoding speed | 12 sentences in 0.43s |
| Separation gap at 64-dim | 0.303 -- beats BGE-large at full 1024-dim |
| Cross-modal image + text alignment | Working |
| Frame integration | (batch, seq, 64) complex64 confirmed |

### REPL Semantic Results (randomly initialized encoder)

```
search "something warm and comforting" fire ocean blanket thunder sunrise
  1. blanket   0.870  <- correctly first
  2. ocean     0.804
  3. fire      0.733
  4. sunrise   0.667
  5. thunder   0.564  <- correctly last

cluster fire ice hot cold dog wolf cat lion
  ice  -> cold  (0.877)  temperature concepts cluster together
  hot  -> cold  (0.925)
  dog  -> lion  (0.911)  animal concepts cluster together
  wolf -> dog   (0.717)

loops "The relationship between gravity and time"
  Loop 1:  0.607  exploring
  Loop 2:  0.489  exploring
  Loop 3:  0.397  exploring
  Loop 4:  0.329  exploring
  Loop 5:  0.276  refining
  Loop 6:  0.256  refining
  Loop 7:  0.238  refining
  Loop 8:  0.000  converging  (stable representation found)
```

These results reflect Qwen3-VL's semantic understanding projected through our complex space. After contrastive pre-training the structure will be sharper.

---

## Research Directions

**Complex Ternary Attention**
Ternary weights ({-1, 0, +1}) applied to complex-valued weights produce 9 discrete values encoding rotations in the complex plane. No multiplication needed -- only addition and subtraction. Hypothesis: complex ternary preserves semantic structure better than real ternary on temporal data. Unpublished combination.

**Phase Coherence Attention**
Tokens with aligned phase interfere constructively. Opposite phase cancels. For temporal data this captures structural information dot-product attention misses entirely.

**FFT Attention on Native Complex Sequences**
O(n log n) vs O(n^2). Our tokens are already complex -- no conversion needed. Novel application to natively complex sequences.

**Adaptive Compute via Convergence Gating**
Loop count scales with problem difficulty. Average inference cost drops ~60% with no accuracy loss.

**Hardware-Aligned Architecture**
Every dimension is a multiple of 16 (d_model=64, k_sparse=16, n_heads=4, ff_dim=256). Aligns perfectly with Blackwell tensor cores and structured sparsity acceleration.

**Self-Modifying Networks**
Long-term target. Canvas environment (Level 4) -> recursive self-improvement with evaluation gate (Level 5). The looping architecture is a natural substrate for self-reasoning.

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
- [x] REPL terminal with encode, similarity, compare, cluster, search, loops
- [x] Semantic results working from randomly initialized encoder
- [ ] Architectural bypass mode for direct-dimension models
- [ ] Contrastive pre-training to >0.85 structure survival correlation
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

RTX 50-series requires PyTorch nightly cu128. Standard stable releases do not support sm_120.

---

## Dependencies

| Library | Purpose |
|---|---|
| torch (nightly cu128) | Core engine |
| sentence-transformers | Qwen3-VL integration |
| datasets | HuggingFace dataset streaming |
| numpy | Data processing |
| pillow | Image generation for vision tests |

Coming soon: openai-whisper, pandas, scikit-learn, plotly

---

*Project Overture -- built from scratch, one layer at a time.*