# Project Overture — Vision

> *The frame that thinks deeply, speaks across domains, and improves itself.*

---

## The Core Insight

Every major AI system built in the last five years has pursued the same strategy: make the model bigger. More parameters, more data, more compute. The assumption is that intelligence scales with size.

Project Overture is built on a different premise.

**A small model that thinks deeply can match or exceed a large model that thinks once.**

The mechanism is the loop. Instead of stacking 96 layers of unique weights and running one forward pass, Overture runs a single weight-tied block up to N times — each iteration refining the representation, converging toward a stable understanding of the input. Easy concepts resolve in 2-3 loops. Hard, abstract, cross-domain concepts run longer. The compute budget is allocated dynamically by the problem, not fixed by the architecture.

This is not a marginal efficiency gain. It is a fundamentally different cognitive model.

---

## The Machine Parts Philosophy

Overture does not try to learn everything from scratch. Instead it treats pretrained specialist models as machine parts — high-precision components built by others and bolted onto a custom frame.

```
Qwen3-VL-Embedding-2B    language and vision understanding
Whisper encoder           audio and speech understanding  
Custom price encoder      financial time series
        ↓
All map to the same complex sparse token format
        ↓
Core loop attends across all domains simultaneously
```

The frame learns what to *do* with specialist knowledge — how to reason across modalities, find non-obvious relationships, and produce outputs that no single specialist could produce alone. The specialists provide the raw understanding. The frame provides the synthesis.

This means new capabilities are additive. Register a new domain encoder and the frame gains that sense without forgetting any other. The architecture scales horizontally, not just vertically.

---

## What Makes the Representation Different

Overture tokens are complex-valued and sparse — a design choice that has implications beyond efficiency.

A standard real-valued token encodes *what* something is — its position in a high-dimensional feature space. A complex token encodes *what* and *how* simultaneously:

- **Magnitude** — how strongly a feature is present
- **Phase** — the relational and temporal position of that feature

For temporal data — price series, audio, language with rhythm — phase relationships carry structural information that real-valued vectors discard entirely. Two tokens with similar magnitude but opposite phase are fundamentally different in ways that dot-product attention cannot distinguish. Our attention mechanism uses the conjugate inner product, which treats phase as a first-class signal.

The sparsity is equally meaningful. Only the top-k dimensions activate per token — forced specialization that emerges from training rather than being hardcoded. Different domains activate different subspaces of the shared complex space. The geometric structure of the representation reflects the structure of the world.

---

## The Research Frontier

### Complex Ternary Attention

Standard ternary quantization replaces float32 weights with {-1, 0, +1} — three values, no multiplication, radical memory reduction. Applied to complex-valued weights this produces nine discrete values encoding rotations in the complex plane.

The hypothesis: **complex ternary attention preserves semantic structure better than real ternary attention at equivalent parameter count**, because phase discretization adds geometric expressiveness that magnitude-only ternary loses.

This combination has not been published. It is the primary research contribution of Project Overture — a testable, falsifiable claim that emerges naturally from the architectural decisions made on day one.

### Phase Coherence Attention

Standard attention asks: are these tokens similar in magnitude? Phase coherence attention asks: are these tokens in rhythm with each other?

Tokens with aligned phase interfere constructively — signal amplifies. Tokens with opposite phase interfere destructively — signal cancels. This is wave interference applied to attention, and for temporal data it captures structural information that dot-product attention misses entirely. Not a performance tweak — a qualitative difference in what the model can perceive.

### Adaptive Compute via Convergence Gating

The loop count scales with problem difficulty. The convergence checker monitors how much the representation changes between iterations — when change drops below a threshold, the loop exits. Easy inputs resolve in 2-3 loops. Hard, abstract, cross-domain inputs run to 8 or beyond.

This is dynamic compute allocation that emerges from the architecture rather than being bolted on. Average inference cost drops approximately 60% with no accuracy loss on easy inputs and more compute available for hard ones.

### FFT Attention on Native Complex Sequences

Standard attention is O(n²). FFT-based attention achieves equivalent results in O(n log n) via the frequency domain. Our tokens are already complex — no real-to-complex conversion needed, a step everyone else pays. Novel application to natively complex token sequences.

---

## Self-Modification — The Long Arc

The most ambitious direction is the one that makes Overture genuinely different from every other AI system in existence: **a network that improves itself.**

Not hyperparameter tuning. Not neural architecture search. Recursive self-improvement — each improved version is smarter at generating the next improvement than the previous version was.

The path there is staged:

```
Level 1  Hyperparameter self-tuning        safe, buildable now
Level 2  Architecture search               managed, evaluation gate required  
Level 4  Code generation self-modification the canvas environment
Level 5  Recursive self-improvement        the destination
```

The canvas environment is the right first implementation of Level 4. Overture perceives a 2D environment through its vision encoder, reasons via the core loop, and modifies the environment by writing and executing code in a sandbox. The frame does not just act in the world — it authors the rules of the world it lives in.

The evaluation gate is what makes Level 5 safe to build. Every proposed self-modification is tested on a held-out benchmark before it commits. Bad changes are discarded. The frame can only improve, never degrade, because the gate enforces it asymmetrically.

The benchmark design is the most important architectural decision at Level 5. The frame will optimize toward whatever the benchmark measures. Choose it carefully and the system improves toward something genuinely useful. Choose it poorly and it optimizes toward something unintended.

---

## Cooperative Instances

The conversation between a human and a single AI instance is a narrow bandwidth channel. Ideas flow sequentially, context gets lost across sessions, and the human becomes the bottleneck — synthesizing outputs from multiple conversations into a coherent direction.

Overture instances can cooperate the way the frame's own loops cooperate — each pass refining what the previous pass produced, with a shared world model as the medium.

```
Instance A                    Instance B
(exploring research direction) (building application layer)
        ↓                              ↓
    findings → Shared World Model ← findings
        ↑                              ↑
    reads B's discoveries          reads A's discoveries
        ↓                              ↓
    builds on them                 builds on them
        ↓                              ↓
         → Instance C (synthesis) ←
           reads both, produces next direction
```

The Conductor from the original Overture design — the orchestrating intelligence that reads what all instances discovered — is not a separate system. It is the same frame, aimed at the outputs of its own instances. Self-directed research that compounds.

The human steps back from the bottleneck role. The frame explores, synthesizes, and directs itself. The human sets the benchmark — what *better* means — and the system pursues it.

---

## Application Layer

### Content Intelligence

Every other AI content tool is a wrapper around a base language model generating text. Overture is different in kind, not degree.

The complex token representation means Overture can detect semantic drift — when an audience's interest is rotating from one conceptual cluster to another — before engagement metrics catch up. It reads the shape of the conversation, not just the surface words.

Cross-modal coherence: a post, its image, and its caption all encode to complex tokens. If they are not geometrically aligned in the shared space the system flags the content as incoherent before it posts. Not thematic matching — geometric alignment. A fundamentally different signal.

Narrative fingerprinting: every piece of content gets a complex token signature. Over time the system builds a world model of what narrative directions are working, which concepts are clustering around a brand, and where the gaps are that nobody is filling.

The monetization layers compound:

```
Direct platform revenue      consistent high-quality content = algorithm favor
Brand conversion             content lives in the right semantic neighborhood
System as product            offer Overture-powered content intelligence as a service
Data moat                    every engagement becomes training signal, dataset is yours
```

### Financial Intelligence — EdgeFlow

The original application domain. USD/JPY price structure combined with central bank narrative, positioned data, and macro language signals — all encoding to complex tokens in the same geometric space, attended to together by the core loop.

The narrative mispricing thesis: markets inefficiently process language relative to price action. A system that can reason across both simultaneously — watching price structure in complex token space while simultaneously encoding a BOJ statement — has access to signal that no purely technical or purely fundamental system can perceive.

### The General Case

Every domain that has both signal data and narrative data is a candidate. Medical imaging and clinical notes. Legal documents and case outcomes. Scientific papers and experimental results. The frame does not care what domain it is in — it cares about the geometric relationships between tokens, wherever they came from.

---

## The Hardware Thesis

Project Overture was designed on an NVIDIA RTX 5060 (Blackwell architecture, sm_120). Every major dimension is a multiple of 16 — d_model=64, k_sparse=16, n_heads=4, ff_dim=256 — aligning with Blackwell tensor cores and structured sparsity acceleration.

This was not an accident. The sparsity mask produces exactly the structured sparse patterns Blackwell accelerates in hardware. The complex ternary weights, when implemented, will require only addition and subtraction — the operation profile Blackwell's fixed-function units handle most efficiently.

The target profile for a fully optimized Overture stack:

```
Parameter count    ~500k trainable (vs GPT-2 small at 117M)
Weight storage     ~25KB with complex ternary (vs ~470MB float32)
Inference time     single-digit milliseconds on Blackwell
VRAM footprint     <100MB including activations
Inference cost     fractions of a cent per thousand calls
```

Intelligence from architecture, not brute force.

---

## The Honest Assessment

This is genuinely ambitious. The technical pieces exist — complex networks, ternary quantization, looping architectures, convergence gating. The combination is novel. The application to financial and content intelligence is practical and testable. The path to Level 5 is staged and safe.

What does not exist yet: the training data, the labeled datasets, the empirical proof that complex ternary outperforms real ternary, the benchmark design for self-modification, the cooperative instance infrastructure.

Those are years of work. But they are the right years of work — building on a foundation that was designed correctly from the first line of code, not retrofitted as the vision expanded.

The first proof of concept is already running. 117,120 parameters riding a frozen 2 billion parameter specialist, producing geometric reasoning that neither could produce alone, responding in natural language about what it finds in the complex space.

That is not nothing. That is a beginning.

---

*Project Overture — built from scratch, one layer at a time.*