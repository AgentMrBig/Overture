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
- **Domain-agnostic shared space** — price data, language, audio, and any other modality map to the same token format

The first application domain is financial time series and narrative — price structure and language signals combined into a unified representation. But the architecture is explicitly designed to be omnicapable.

---

## Architecture

```
ANY INPUT (price, text, audio, ...)
        ↓
Domain Encoder          — lightweight, domain-specific
                          maps raw input → complex sparse token
        ↓
Complex Sparse Token    — shape: (batch, seq_len, d_model) complex64
                          magnitude: feature presence
                          phase:     relational position
                          sparse:    top-k dims active per token
        ↓
┌─── CORE LOOP (weight-tied) ──────────────────────┐
│                                                   │
│   Complex Multi-Head Attention                    │
│   scores = Re(Q · K†) / √d  ← conjugate inner   │
│        ↓                       product           │
│   Complex Feed-Forward                            │
│   modReLU — thresholds magnitude, keeps phase     │
│        ↓                                          │
│   Convergence Check                               │
│   exit early if representation stabilized         │
│        ↓                                          │
│   Loop again (same weights) or exit               │
└───────────────────────────────────────────────────┘
        ↓
Output Head             — pooling → complex→real → projection
                          binary / multiclass / regression
        ↓
PREDICTION
```

### Key Numbers (default config)

| Component | Parameters |
|---|---|
| Domain Encoder (per domain) | ~10,000 |
| Core Loop (shared, all iterations) | ~100,000 |
| Output Head | ~12,000 |
| **Total** | **~123,000** |

GPT-2 small has 117,000,000 parameters. Overture has 123,000 and achieves depth through iteration.

---

## Token Format

Every input domain maps to the same token format — a **complex-valued sparse vector**:

```python
# Price candle (12 features) → complex token
price_token    shape: (batch, seq_len, 64)  dtype: complex64

# News headline (384-dim ST embedding) → complex token  
language_token shape: (batch, seq_len, 64)  dtype: complex64

# Both live in the same space — concatenate and attend
combined       shape: (batch, seq_len+1, 64)
```

This is the core architectural insight. The Transformer does not care whether a token came from price data or text. It sees a sequence of complex vectors and finds relationships between them.

---

## Files

```
Overture/
├── token_encoder.py      # Domain encoders → complex sparse tokens
│                         # DomainProjection, ComplexLift (polar),
│                         # SparsityMask, ComplexPositionalEncoding,
│                         # DomainRegistry
│
├── core_loop.py          # Weight-tied iterative Transformer
│                         # ComplexLinear, ComplexMultiHeadAttention,
│                         # ModReLU, ComplexFeedForward,
│                         # ConvergenceChecker, CoreLoop, OvertureModel
│
├── output_head.py        # Task-specific output heads
│                         # ComplexPooling, ComplexToReal, OutputHead,
│                         # OvertureWithHead
│
├── training_loop.py      # Full training pipeline
│                         # SyntheticSequenceDataset, train(),
│                         # evaluate(), AdamW + cosine LR
│
└── language_demo.py      # Language encoder demonstration
                          # sentence-transformers integration,
                          # semantic similarity in shared space,
                          # combined price + language sequences
```

---

## Quickstart

**1. Create and activate virtual environment**
```bash
python -m venv overture_env
source overture_env/Scripts/activate   # Windows Git Bash
# or
source overture_env/bin/activate       # Mac / Linux
```

**2. Install dependencies**
```bash
# PyTorch — match your CUDA version (check with nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Other dependencies
pip install sentence-transformers numpy
```

**3. Run the pipeline in order**
```bash
# Verify token encoder
python token_encoder.py

# Verify core loop
python core_loop.py

# Verify output head
python output_head.py

# Run first training task
python training_loop.py

# Run language encoder demo
python language_demo.py
```

---

## Results So Far

### Synthetic Trend Classification
Training the full pipeline on a synthetic temporal pattern detection task:

| Metric | Value |
|---|---|
| Task | Detect uptrend vs downtrend in 60-step sequence |
| Parameters | 123,138 |
| Epochs to convergence | 5 |
| Final val accuracy | 100.0% |
| Prediction confidence | 99.8 – 99.9% |
| Hardware | NVIDIA RTX 5060 |

### Language Encoder Semantic Structure
Semantic similarity matrix in shared complex token space (untrained encoder):

| | Hawkish | Dovish | Risk-off | Risk-on |
|---|---|---|---|---|
| **Hawkish** | 1.000 | 0.786 | 0.601 | 0.793 |
| **Dovish** | 0.786 | 1.000 | 0.720 | 0.766 |
| **Risk-off** | 0.601 | 0.720 | 1.000 | 0.573 |
| **Risk-on** | 0.793 | 0.766 | 0.573 | 1.000 |

Semantic structure from sentence-transformers is preserved through the projection into our 64-dim complex space. Risk-off vs Risk-on (0.573) is the most separated pair — as expected.

---

## Design Decisions

**Why complex-valued tokens?**
A complex number has magnitude and phase. Magnitude encodes how strongly a feature is present. Phase encodes relational/temporal position. For any signal data — price, audio, language — phase relationships carry information that real-valued vectors discard. Complex multiplication is also a natural way to encode rotation and transformation, making positional encoding geometrically meaningful.

**Why sparse?**
Different inputs should activate different subspaces of the representation. A high-volatility candle during a news event should light up different dimensions than a quiet Asian session candle. Learned top-k sparsity forces this specialization to emerge from training rather than being hardcoded.

**Why weight-tied loops instead of deep layers?**
A 96-layer Transformer has 96 sets of unique weights. A looped Transformer with 1 block run 96 times has 1 set of weights. The looped model is forced to learn weights that are useful at every stage of refinement — pass 1 AND pass 50. This acts as a powerful regularizer and dramatically reduces parameter count while preserving computational depth. The convergence checker allows adaptive compute — easy inputs exit early, hard inputs run more loops.

**Why domain-agnostic?**
The goal is not a trading model or a language model. The goal is an architecture that can be aimed at any problem domain by registering a new encoder. Price, language, audio, sensor data, biological sequences — all project into the same shared space and are processed by the same core loop.

---

## Roadmap

- [x] Complex sparse token encoder
- [x] Weight-tied iterative core loop
- [x] Convergence detection
- [x] Multi-task output head
- [x] Training pipeline with walk-forward support
- [x] Language encoder via sentence-transformers
- [x] Combined price + language token sequences
- [ ] Real OHLCV price data pipeline
- [ ] Combined price + language training task
- [ ] Contrastive pre-training for language encoder
- [ ] Multi-timeframe price encoding
- [ ] Walk-forward validation on real data
- [ ] Signal bridge to EdgeFlow Trader (MT4)
- [ ] Regime detection head
- [ ] Visualization dashboard for attention across loops

---

## Dependencies

| Library | Purpose |
|---|---|
| `torch` | Core engine — all tensor math and training |
| `sentence-transformers` | Pretrained text → 384-dim semantic vectors |
| `numpy` | Data processing |

Optional (for data pipeline, coming soon):
| Library | Purpose |
|---|---|
| `pandas` | CSV loading and feature engineering |
| `scikit-learn` | Normalization, walk-forward splits |
| `plotly` | Training visualization |

---

## Notes on Hardware

Developed and tested on:
- **GPU**: NVIDIA GeForce RTX 5060 (Blackwell, sm_120)
- **CUDA**: 13.1
- **PyTorch**: nightly cu128 build (required for sm_120 support on Windows)
- **Python**: 3.10.6

If you're on an older GPU (RTX 30xx or 40xx series), the standard stable PyTorch release will work fine with the appropriate cu118/cu121/cu124 index URL.

---

*Project Overture — built from scratch, one layer at a time.*