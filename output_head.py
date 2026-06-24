"""
Omnicapable Transformer — Output Head
======================================
Reads the final complex token representations from the
core loop and maps them to task-specific predictions.

Three stages:
    1. Pooling      — collapse (batch, seq_len, d_model) → (batch, d_model)
    2. Complex→Real — extract real-valued signal from complex representation
    3. Projection   — map to task output size

Supports multiple task types:
    'binary'         — single sigmoid output (e.g. up/down, spam/not spam)
    'multiclass'     — softmax over N classes (e.g. topic classification)
    'regression'     — unbounded scalar or vector (e.g. price return)
    'sequence'       — per-token output (e.g. token labeling)

Usage:
    head = OutputHead(d_model=64, task='binary')
    logits = head(result['output'])   # (batch,) for binary

    # Or use full OvertureModel with head attached:
    model = OvertureModel(d_model=64, ...)
    model.register_domain('price', input_dim=12)
    model.attach_head(task='binary')

    pred = model.predict('price', x)  # (batch,) sigmoid probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core_loop import OvertureModel


# ─────────────────────────────────────────────────────────
#  1. POOLING STRATEGIES
#     Collapse the sequence dimension into one vector.
#     Different strategies capture different things.
# ─────────────────────────────────────────────────────────

class ComplexPooling(nn.Module):
    """
    Pools a sequence of complex tokens into a single vector.

    Strategies:
        'mean'    — average all tokens (good for classification)
        'max'     — max magnitude per dim (good for detecting features)
        'last'    — use the last token (good for sequences with order)
        'attention'— learned weighted average (most expressive)

    Args:
        d_model   : token dimension
        strategy  : pooling strategy
    """
    def __init__(self, d_model: int, strategy: str = 'attention'):
        super().__init__()
        assert strategy in ('mean', 'max', 'last', 'attention')
        self.strategy = strategy
        self.d_model  = d_model

        if strategy == 'attention':
            # Learned scoring — some tokens matter more than others
            # Score based on magnitude (how "active" is this token)
            self.scorer = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z:   (batch, seq_len, d_model) complex
        out: (batch, d_model) complex — pooled representation
        """
        if self.strategy == 'mean':
            return z.mean(dim=1)

        elif self.strategy == 'max':
            # Max over magnitude, preserving complex value
            magnitudes = z.abs()                        # (B, S, D) real
            idx = magnitudes.argmax(dim=1, keepdim=True)# (B, 1, D)
            return z.gather(1, idx.expand_as(z)).squeeze(1)

        elif self.strategy == 'last':
            return z[:, -1, :]

        elif self.strategy == 'attention':
            # Score each token by its magnitude profile
            magnitudes = z.abs()                        # (B, S, D) real
            scores = self.scorer(magnitudes)            # (B, S, 1) real
            weights = F.softmax(scores, dim=1)          # (B, S, 1) normalized
            # Weighted sum — weights are real, z is complex
            weights_c = weights.to(torch.complex64)
            pooled = (weights_c * z).sum(dim=1)         # (B, D) complex
            return pooled


# ─────────────────────────────────────────────────────────
#  2. COMPLEX → REAL EXTRACTION
#     The core loop outputs complex tensors.
#     We need real-valued predictions for most tasks.
#     Four ways to extract real signal from complex tensors,
#     each preserving different information.
# ─────────────────────────────────────────────────────────

class ComplexToReal(nn.Module):
    """
    Extracts a real-valued representation from complex tokens.

    Modes:
        'magnitude'  — |z| per dim. Discards phase, keeps energy.
                       Good when phase is noise, magnitude is signal.

        'concat'     — [real, imag] concatenated. Keeps everything.
                       Output dim = 2 * d_model.
                       Good when both components carry signal.

        'real'       — just the real part. Fast, simple.
                       Good when imaginary part is auxiliary.

        'learned'    — small network decides how to mix real/imag.
                       Most flexible, learns the best extraction.
                       Output dim = d_model.

    Args:
        d_model : complex token dimension
        mode    : extraction mode
    """
    def __init__(self, d_model: int, mode: str = 'learned'):
        super().__init__()
        assert mode in ('magnitude', 'concat', 'real', 'learned')
        self.mode    = mode
        self.d_model = d_model

        if mode == 'learned':
            # Takes [real, imag] and learns the best combination
            self.mixer = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

    def output_dim(self) -> int:
        """Returns the output dimension after extraction."""
        if self.mode == 'concat':
            return self.d_model * 2
        return self.d_model

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z:   (..., d_model) complex
        out: (..., output_dim) real
        """
        if self.mode == 'magnitude':
            return z.abs()

        elif self.mode == 'real':
            return z.real

        elif self.mode == 'concat':
            return torch.cat([z.real, z.imag], dim=-1)

        elif self.mode == 'learned':
            combined = torch.cat([z.real, z.imag], dim=-1)
            return self.mixer(combined)


# ─────────────────────────────────────────────────────────
#  3. OUTPUT HEAD
#     Final projection from real-valued representation
#     to task-specific output.
# ─────────────────────────────────────────────────────────

class OutputHead(nn.Module):
    """
    Task-specific output head.

    Chains: ComplexPooling → ComplexToReal → Linear projection → activation

    Args:
        d_model       : input complex token dimension
        task          : 'binary', 'multiclass', 'regression', 'sequence'
        n_classes     : number of classes (for multiclass)
        output_dim    : output size (for regression/sequence)
        pool_strategy : pooling strategy
        extract_mode  : complex→real extraction mode
        dropout       : dropout before final projection
    """
    def __init__(
        self,
        d_model       : int   = 64,
        task          : str   = 'binary',
        n_classes     : int   = 2,
        output_dim    : int   = 1,
        pool_strategy : str   = 'attention',
        extract_mode  : str   = 'learned',
        dropout       : float = 0.1,
    ):
        super().__init__()
        self.task      = task
        self.d_model   = d_model

        # Sequence tasks don't pool — they output per token
        self.needs_pool = (task != 'sequence')

        if self.needs_pool:
            self.pool = ComplexPooling(d_model, strategy=pool_strategy)

        self.extractor = ComplexToReal(d_model, mode=extract_mode)
        real_dim       = self.extractor.output_dim()

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(real_dim)

        # Task-specific projection
        if task == 'binary':
            self.proj = nn.Linear(real_dim, 1)

        elif task == 'multiclass':
            self.proj = nn.Linear(real_dim, n_classes)

        elif task == 'regression':
            self.proj = nn.Sequential(
                nn.Linear(real_dim, real_dim // 2),
                nn.GELU(),
                nn.Linear(real_dim // 2, output_dim),
            )

        elif task == 'sequence':
            # Per-token output — no pooling
            self.proj = nn.Linear(real_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z:   (batch, seq_len, d_model) complex  — core loop output
        out: task-dependent real tensor

            binary     → (batch,)         sigmoid probabilities
            multiclass → (batch, n_classes) logits
            regression → (batch, output_dim)
            sequence   → (batch, seq_len, output_dim)
        """
        # Pool sequence → single vector (except sequence tasks)
        if self.needs_pool:
            z = self.pool(z)            # (batch, d_model) complex

        # Extract real signal
        x = self.extractor(z)          # (batch, real_dim) or (batch, seq, real_dim)

        # Normalize + dropout
        x = self.norm(x)
        x = self.dropout(x)

        # Project to output
        out = self.proj(x)

        # Apply task activation
        if self.task == 'binary':
            return torch.sigmoid(out.squeeze(-1))   # (batch,)

        elif self.task == 'multiclass':
            return out                               # (batch, n_classes) raw logits

        elif self.task == 'regression':
            return out                               # (batch, output_dim)

        elif self.task == 'sequence':
            return out                               # (batch, seq_len, output_dim)


# ─────────────────────────────────────────────────────────
#  4. ATTACH HEAD TO OVERTURE MODEL
#     Extends OvertureModel with a predict() method
#     that chains encoder → core loop → output head.
# ─────────────────────────────────────────────────────────

class OvertureWithHead(nn.Module):
    """
    Full Omnicapable Transformer with output head attached.

    This is the trainable end-to-end model:
        raw input → tokens → core loop → output head → prediction

    Args:
        d_model, k_sparse, n_heads, max_loops — core architecture
        task, n_classes, output_dim          — task configuration
        pool_strategy, extract_mode          — head configuration
    """
    def __init__(
        self,
        d_model       : int   = 64,
        k_sparse      : int   = 16,
        n_heads       : int   = 4,
        max_loops     : int   = 8,
        ff_multiplier : int   = 4,
        dropout       : float = 0.1,
        task          : str   = 'binary',
        n_classes     : int   = 2,
        output_dim    : int   = 1,
        pool_strategy : str   = 'attention',
        extract_mode  : str   = 'learned',
    ):
        super().__init__()

        self.backbone = OvertureModel(
            d_model       = d_model,
            k_sparse      = k_sparse,
            n_heads       = n_heads,
            max_loops     = max_loops,
            ff_multiplier = ff_multiplier,
            dropout       = dropout,
        )

        self.head = OutputHead(
            d_model       = d_model,
            task          = task,
            n_classes     = n_classes,
            output_dim    = output_dim,
            pool_strategy = pool_strategy,
            extract_mode  = extract_mode,
            dropout       = dropout,
        )

    def register_domain(self, name: str, input_dim: int, lift_mode: str = 'polar'):
        self.backbone.register_domain(name, input_dim, lift_mode)

    def forward(
        self,
        domain         : str,
        x              : torch.Tensor,
        return_history : bool = False,
    ) -> dict:
        """
        Full forward pass.

        Returns dict with:
            'prediction'   : task output
            'output'       : raw complex representations
            'loops_run'    : iterations used
            'converged'    : early exit triggered
            'attn_history' : attention per loop
        """
        result = self.backbone(domain, x, return_history=return_history)
        prediction = self.head(result['output'])
        result['prediction'] = prediction
        return result

    def count_parameters(self) -> dict:
        backbone = self.backbone.count_parameters()
        head_params = sum(p.numel() for p in self.head.parameters())
        return {
            'encoder'   : backbone['encoder'],
            'core_loop' : backbone['core_loop'],
            'head'      : head_params,
            'total'     : backbone['total'] + head_params,
        }


# ─────────────────────────────────────────────────────────
#  SMOKE TEST
#  python output_head.py
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    BATCH   = 4
    SEQ_LEN = 60

    # ── Test 1: Binary classification (e.g. price up/down) ──
    print("Test 1: Binary classification")
    model = OvertureWithHead(
        d_model=64, k_sparse=16, n_heads=4, max_loops=8,
        task='binary'
    )
    model.register_domain('price', input_dim=12)
    model.to(device)
    model.eval()

    x = torch.randn(BATCH, SEQ_LEN, 12).to(device)
    with torch.no_grad():
        result = model('price', x)

    print(f"  Input shape      : {tuple(x.shape)}")
    print(f"  Prediction shape : {tuple(result['prediction'].shape)}")
    print(f"  Predictions      : {result['prediction'].cpu().numpy().round(4)}")
    print(f"  Loops run        : {result['loops_run']}")
    print(f"  Range check      : all in [0,1] = "
          f"{bool((result['prediction'] >= 0).all() and (result['prediction'] <= 1).all())}")
    print()

    # ── Test 2: Multiclass (e.g. market regime) ──
    print("Test 2: Multiclass — 4 market regimes")
    model2 = OvertureWithHead(
        d_model=64, k_sparse=16, n_heads=4, max_loops=8,
        task='multiclass', n_classes=4
    )
    model2.register_domain('price', input_dim=12)
    model2.to(device)
    model2.eval()

    with torch.no_grad():
        result2 = model2('price', x)

    print(f"  Prediction shape : {tuple(result2['prediction'].shape)}")
    print(f"  Logits sample    : {result2['prediction'][0].cpu().numpy().round(4)}")
    print(f"  Predicted class  : {result2['prediction'].argmax(dim=-1).cpu().numpy()}")
    print()

    # ── Test 3: Regression (e.g. next N pip return) ──
    print("Test 3: Regression — predict pip return")
    model3 = OvertureWithHead(
        d_model=64, k_sparse=16, n_heads=4, max_loops=8,
        task='regression', output_dim=1
    )
    model3.register_domain('price', input_dim=12)
    model3.to(device)
    model3.eval()

    with torch.no_grad():
        result3 = model3('price', x)

    print(f"  Prediction shape : {tuple(result3['prediction'].shape)}")
    print(f"  Predictions      : {result3['prediction'].squeeze().cpu().numpy().round(4)}")
    print()

    # ── Full parameter count ──
    params = model.count_parameters()
    print(f"{'─'*50}")
    print(f"Full model parameter counts (binary head):")
    print(f"  Encoder    : {params['encoder']:>10,}")
    print(f"  Core loop  : {params['core_loop']:>10,}")
    print(f"  Head       : {params['head']:>10,}")
    print(f"  ─────────────────────")
    print(f"  Total      : {params['total']:>10,}")
    print(f"{'─'*50}")
    print(f"\nFull pipeline ready.")
    print(f"Next: training loop.")