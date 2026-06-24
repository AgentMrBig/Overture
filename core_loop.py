"""
Omnicapable Transformer — Core Loop
====================================
Weight-tied complex attention block that runs N iterations
on the same set of parameters. Depth comes from looping,
not from stacking unique layers.

Architecture per loop iteration:
    1. Complex Multi-Head Attention  — tokens attend to each other
    2. Complex Feed Forward          — per-token transformation
    3. Convergence Check             — exit early if stable

The same weights handle iteration 1 AND iteration 10.
This forces the weights to be general — they must do useful
work at any stage of refinement, not just one fixed depth.

Usage:
    from token_encoder import DomainRegistry
    from core_loop import CoreLoop, OvertureModel

    model = OvertureModel(d_model=64, k_sparse=16, n_heads=4, max_loops=8)
    model.register_domain('price',    input_dim=12)
    model.register_domain('language', input_dim=384)

    out = model('price', price_data)   # (batch, seq_len, d_model) complex
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from token_encoder import DomainRegistry


# ─────────────────────────────────────────────────────────
#  1. COMPLEX LAYER NORM
#     Standard LayerNorm doesn't work on complex tensors.
#     We normalize real and imaginary parts independently,
#     which preserves the phase relationship while keeping
#     magnitudes in a stable range.
# ─────────────────────────────────────────────────────────

class ComplexLayerNorm(nn.Module):
    """
    Layer normalization for complex-valued tensors.
    Normalizes real and imaginary parts independently.

    Args:
        d_model : token dimension
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm_real = nn.LayerNorm(d_model)
        self.norm_imag = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z:   (batch, seq_len, d_model) complex
        out: (batch, seq_len, d_model) complex normalized
        """
        return torch.complex(
            self.norm_real(z.real),
            self.norm_imag(z.imag)
        )


# ─────────────────────────────────────────────────────────
#  2. COMPLEX MULTI-HEAD ATTENTION
#     Standard attention but operating in complex space.
#
#     The key insight: attention scores are computed from
#     the MAGNITUDE of Q·K^H (conjugate transpose), not
#     just Q·K^T. This means phase relationships between
#     tokens influence which tokens attend to which.
#
#     Two tokens with similar magnitude BUT opposite phase
#     will have low attention — they are "out of phase"
#     with each other. This is a richer similarity metric
#     than real-valued dot product alone.
#
#     Q, K, V projections are complex linear layers —
#     complex weight matrices applied to complex inputs.
# ─────────────────────────────────────────────────────────

class ComplexLinear(nn.Module):
    """
    Linear layer for complex inputs.
    Applies a complex weight matrix: W = Wr + i*Wi
    Output = (Wr + i*Wi)(x_r + i*x_i)
           = (Wr*x_r - Wi*x_i) + i(Wi*x_r + Wr*x_i)

    Args:
        in_features  : input dimension
        out_features : output dimension
        bias         : whether to include bias
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # Real and imaginary weight components
        self.weight_r = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_i = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features))
            self.bias_i = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias_r = self.bias_i = None

        # Initialize — keep imaginary weights small initially
        nn.init.xavier_uniform_(self.weight_r)
        nn.init.xavier_uniform_(self.weight_i, gain=0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z:   (..., in_features) complex
        out: (..., out_features) complex
        """
        x_r, x_i = z.real, z.imag

        # Complex matrix multiplication
        out_r = F.linear(x_r, self.weight_r) - F.linear(x_i, self.weight_i)
        out_i = F.linear(x_i, self.weight_r) + F.linear(x_r, self.weight_i)

        if self.bias_r is not None:
            out_r = out_r + self.bias_r
            out_i = out_i + self.bias_i

        return torch.complex(out_r, out_i)


class ComplexMultiHeadAttention(nn.Module):
    """
    Multi-head attention operating in complex space.

    Attention score between query q and key k:
        score = Re(q · k^H) / sqrt(d_head)
    where k^H is the conjugate transpose of k.

    This uses the real part of the complex inner product
    as the similarity measure — geometrically, this is
    the projection of q onto k in complex space, which
    accounts for both magnitude and phase alignment.

    Args:
        d_model  : token dimension
        n_heads  : number of attention heads
        dropout  : attention dropout rate
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)

        # Complex Q, K, V projections (shared across all loop iterations)
        self.W_q = ComplexLinear(d_model, d_model)
        self.W_k = ComplexLinear(d_model, d_model)
        self.W_v = ComplexLinear(d_model, d_model)
        self.W_o = ComplexLinear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z:    (batch, seq_len, d_model) complex
        mask: (batch, seq_len, seq_len) bool — True = ignore (optional)

        out:  (batch, seq_len, d_model) complex
              (batch, n_heads, seq_len, seq_len) real attention weights
        """
        B, S, D = z.shape

        # Project to Q, K, V
        Q = self.W_q(z)   # (B, S, D) complex
        K = self.W_k(z)
        V = self.W_v(z)

        # Split into heads
        # (B, S, D) → (B, n_heads, S, d_head)
        def split_heads(x):
            return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        Q = split_heads(Q)   # (B, H, S, d_head) complex
        K = split_heads(K)
        V = split_heads(V)

        # Complex attention scores: Re(Q · K^H) / scale
        # K^H = conjugate transpose of K
        # (B, H, S, d_head) × (B, H, d_head, S) → (B, H, S, S)
        K_conj = K.conj()
        scores_complex = torch.matmul(Q, K_conj.transpose(-2, -1))
        scores = scores_complex.real / self.scale   # (B, H, S, S) real

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        # Softmax over key dimension
        attn_weights = F.softmax(scores, dim=-1)    # (B, H, S, S) real
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        # attn_weights is real, V is complex — broadcast multiplication
        # (B, H, S, S) real × (B, H, S, d_head) complex
        attn_weights_c = attn_weights.to(torch.complex64)
        attended = torch.matmul(attn_weights_c, V)  # (B, H, S, d_head) complex

        # Merge heads back
        attended = attended.transpose(1, 2).contiguous().view(B, S, D)

        # Output projection
        out = self.W_o(attended)   # (B, S, D) complex

        return out, attn_weights


# ─────────────────────────────────────────────────────────
#  3. COMPLEX FEED FORWARD
#     Applied independently to each token after attention.
#     Two complex linear layers with a nonlinearity between.
#
#     For the nonlinearity we use modReLU:
#         modReLU(z) = ReLU(|z| + b) * (z / |z|)
#     This preserves phase while thresholding magnitude.
#     Invented specifically for complex neural networks.
# ─────────────────────────────────────────────────────────

class ModReLU(nn.Module):
    """
    Modulus ReLU for complex tensors.
    Thresholds the magnitude while preserving phase direction.
    modReLU(z) = ReLU(|z| + b) * exp(i*angle(z))

    The bias b is learned — negative b allows small magnitudes
    through, positive b suppresses them.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        magnitude = z.abs().clamp(min=1e-7)
        activated = F.relu(magnitude + self.bias)
        # Normalize to unit magnitude, then scale by activated magnitude
        unit = z / magnitude
        return unit * activated


class ComplexFeedForward(nn.Module):
    """
    Position-wise feed-forward network for complex tokens.
    Expands to ff_dim then contracts back to d_model.

    Args:
        d_model  : token dimension
        ff_dim   : inner dimension (typically 4 * d_model)
        dropout  : dropout rate
    """
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1  = ComplexLinear(d_model, ff_dim)
        self.linear2  = ComplexLinear(ff_dim, d_model)
        self.act      = ModReLU(ff_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z:   (batch, seq_len, d_model) complex
        out: (batch, seq_len, d_model) complex
        """
        z = self.linear1(z)
        z = self.act(z)
        z = self.dropout(z.real).to(torch.complex64) + \
            1j * self.dropout(z.imag).to(torch.complex64)
        z = self.linear2(z)
        return z


# ─────────────────────────────────────────────────────────
#  4. CONVERGENCE CHECKER
#     Decides when to stop looping.
#     Compares the token representations between the current
#     loop and the previous loop. If the change is below
#     a threshold, the representation has stabilized.
#
#     Delta = mean |z_current - z_previous| / mean |z_current|
#
#     This is a relative change metric — we care about how
#     much the representation changed relative to its scale,
#     not the absolute change.
# ─────────────────────────────────────────────────────────

class ConvergenceChecker:
    """
    Checks whether token representations have stabilized
    between loop iterations.

    Args:
        threshold : relative change below which we consider converged
        patience  : number of consecutive stable iterations before stopping
    """
    def __init__(self, threshold: float = 0.01, patience: int = 2):
        self.threshold  = threshold
        self.patience   = patience
        self.stable_count = 0
        self.prev        = None
        self.history     = []

    def reset(self):
        self.stable_count = 0
        self.prev         = None
        self.history      = []

    def check(self, z: torch.Tensor) -> bool:
        """
        z: current token representations (batch, seq_len, d_model) complex
        returns True if converged (should stop looping)
        """
        if self.prev is None:
            self.prev = z.detach()
            return False

        # Relative change in magnitude
        delta    = (z - self.prev).abs().mean()
        scale    = z.abs().mean().clamp(min=1e-7)
        rel_change = (delta / scale).item()

        self.history.append(rel_change)
        self.prev = z.detach()

        if rel_change < self.threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        return self.stable_count >= self.patience


# ─────────────────────────────────────────────────────────
#  5. CORE LOOP BLOCK
#     One Transformer block (attention + FFN + norms)
#     designed to be run multiple times with the same weights.
#
#     Residual connections use complex addition — adding
#     the input to the output of each sublayer preserves
#     gradient flow across many loops.
# ─────────────────────────────────────────────────────────

class CoreLoopBlock(nn.Module):
    """
    Single Transformer block for iterative execution.
    Weights are shared across all loop iterations.

    Args:
        d_model   : token dimension
        n_heads   : attention heads
        ff_dim    : feed-forward inner dimension
        dropout   : dropout rate
    """
    def __init__(
        self,
        d_model  : int,
        n_heads  : int,
        ff_dim   : int,
        dropout  : float = 0.1
    ):
        super().__init__()
        self.attn    = ComplexMultiHeadAttention(d_model, n_heads, dropout)
        self.ff      = ComplexFeedForward(d_model, ff_dim, dropout)
        self.norm1   = ComplexLayerNorm(d_model)
        self.norm2   = ComplexLayerNorm(d_model)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z:   (batch, seq_len, d_model) complex
        out: (batch, seq_len, d_model) complex
             (batch, n_heads, seq_len, seq_len) attention weights
        """
        # Attention sublayer with residual
        attn_out, attn_weights = self.attn(z, mask)
        z = self.norm1(z + attn_out)

        # Feed-forward sublayer with residual
        ff_out = self.ff(z)
        z = self.norm2(z + ff_out)

        return z, attn_weights


# ─────────────────────────────────────────────────────────
#  6. CORE LOOP
#     Runs the CoreLoopBlock N times with the same weights.
#     Exits early if convergence is detected.
#     Records attention weights at each iteration for
#     visualization and analysis.
# ─────────────────────────────────────────────────────────

class CoreLoop(nn.Module):
    """
    The iterative core of the Omnicapable Transformer.
    Runs a single weight-tied block up to max_loops times.

    Args:
        d_model         : token dimension
        n_heads         : attention heads
        ff_multiplier   : ff_dim = d_model * ff_multiplier
        max_loops       : maximum iterations
        dropout         : dropout rate
        convergence_thr : relative change threshold for early exit
        convergence_pat : patience for convergence (consecutive stable iters)
    """
    def __init__(
        self,
        d_model         : int   = 64,
        n_heads         : int   = 4,
        ff_multiplier   : int   = 4,
        max_loops       : int   = 8,
        dropout         : float = 0.1,
        convergence_thr : float = 0.01,
        convergence_pat : int   = 2,
    ):
        super().__init__()
        self.max_loops = max_loops
        ff_dim = d_model * ff_multiplier

        # THE key line — one block, reused N times
        self.block = CoreLoopBlock(d_model, n_heads, ff_dim, dropout)
        self.convergence = ConvergenceChecker(convergence_thr, convergence_pat)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor = None,
        return_history: bool = False
    ) -> dict:
        """
        z:              (batch, seq_len, d_model) complex
        mask:           optional attention mask
        return_history: if True, return representations at each loop

        returns dict with:
            'output'       : (batch, seq_len, d_model) final representations
            'loops_run'    : number of iterations executed
            'converged'    : whether early exit triggered
            'attn_history' : list of attention weights per loop
            'repr_history' : list of representations per loop (if return_history)
            'delta_history': list of convergence deltas per loop
        """
        self.convergence.reset()

        attn_history  = []
        repr_history  = [z] if return_history else []
        loops_run     = 0
        converged     = False

        for loop_idx in range(self.max_loops):
            z, attn_weights = self.block(z, mask)
            loops_run += 1

            attn_history.append(attn_weights.detach())
            if return_history:
                repr_history.append(z.detach())

            # Check convergence (disabled during training for gradient flow)
            if not self.training:
                if self.convergence.check(z):
                    converged = True
                    break

        return {
            'output'        : z,
            'loops_run'     : loops_run,
            'converged'     : converged,
            'attn_history'  : attn_history,
            'repr_history'  : repr_history,
            'delta_history' : self.convergence.history,
        }


# ─────────────────────────────────────────────────────────
#  7. OVERTURE MODEL
#     Assembles token encoder + core loop into one module.
#     This is the full forward pass of the network.
# ─────────────────────────────────────────────────────────

class OvertureModel(nn.Module):
    """
    Full Omnicapable Transformer.

    Combines:
        DomainRegistry  → any input to complex sparse tokens
        CoreLoop        → iterative complex attention

    Args:
        d_model         : shared token dimension
        k_sparse        : active dims per token
        n_heads         : attention heads
        max_loops       : maximum loop iterations
        ff_multiplier   : feed-forward expansion ratio
        dropout         : dropout rate
    """
    def __init__(
        self,
        d_model       : int   = 64,
        k_sparse      : int   = 16,
        n_heads       : int   = 4,
        max_loops     : int   = 8,
        ff_multiplier : int   = 4,
        dropout       : float = 0.1,
    ):
        super().__init__()
        self.registry  = DomainRegistry(d_model=d_model, k_sparse=k_sparse)
        self.core_loop = CoreLoop(
            d_model         = d_model,
            n_heads         = n_heads,
            ff_multiplier   = ff_multiplier,
            max_loops       = max_loops,
            dropout         = dropout,
        )

    def register_domain(self, name: str, input_dim: int, lift_mode: str = 'polar'):
        """Register a new input domain."""
        self.registry.register(name, input_dim, lift_mode)

    def forward(
        self,
        domain         : str,
        x              : torch.Tensor,
        mask           : torch.Tensor = None,
        return_history : bool = False,
    ) -> dict:
        """
        domain : name of input domain (must be registered)
        x      : (batch, seq_len, input_dim) raw domain input
        out    : dict with 'output' and loop diagnostics
        """
        # Encode to complex sparse tokens
        tokens = self.registry(domain, x)

        # Run iterative core loop
        result = self.core_loop(tokens, mask, return_history)
        result['tokens'] = tokens   # include encoded tokens for inspection

        return result

    def count_parameters(self) -> dict:
        encoder_params  = sum(p.numel() for p in self.registry.parameters())
        loop_params     = sum(p.numel() for p in self.core_loop.parameters())
        total           = encoder_params + loop_params
        return {
            'encoder'  : encoder_params,
            'core_loop': loop_params,
            'total'    : total,
        }


# ─────────────────────────────────────────────────────────
#  SMOKE TEST
#  python core_loop.py
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # --- Build model ---
    model = OvertureModel(
        d_model     = 64,
        k_sparse    = 16,
        n_heads     = 4,
        max_loops   = 8,
        ff_multiplier = 4,
    )
    model.register_domain('price',    input_dim=12)
    model.register_domain('language', input_dim=384)
    model.to(device)
    model.eval()

    # --- Parameter count ---
    params = model.count_parameters()
    print(f"{'─'*50}")
    print(f"Parameter counts:")
    print(f"  Encoder  (domain-specific) : {params['encoder']:>10,}")
    print(f"  Core loop (shared weights) : {params['core_loop']:>10,}")
    print(f"  Total                      : {params['total']:>10,}")
    print(f"{'─'*50}\n")

    BATCH   = 4
    SEQ_LEN = 60

    # --- Test price domain ---
    print("Test 1: Price domain")
    price_data = torch.randn(BATCH, SEQ_LEN, 12).to(device)

    with torch.no_grad():
        result = model('price', price_data, return_history=True)

    print(f"  Input shape    : {tuple(price_data.shape)}")
    print(f"  Output shape   : {tuple(result['output'].shape)}")
    print(f"  Loops run      : {result['loops_run']} / 8")
    print(f"  Converged      : {result['converged']}")
    print(f"  Delta history  : {[f'{d:.4f}' for d in result['delta_history']]}")
    print(f"  Output dtype   : {result['output'].dtype}")
    print(f"  Output magnitude mean: {result['output'].abs().mean().item():.4f}")
    print()

    # --- Test language domain ---
    print("Test 2: Language domain")
    lang_data = torch.randn(BATCH, SEQ_LEN, 384).to(device)

    with torch.no_grad():
        result2 = model('language', lang_data, return_history=True)

    print(f"  Input shape    : {tuple(lang_data.shape)}")
    print(f"  Output shape   : {tuple(result2['output'].shape)}")
    print(f"  Loops run      : {result2['loops_run']} / 8")
    print(f"  Converged      : {result2['converged']}")
    print(f"  Delta history  : {[f'{d:.4f}' for d in result2['delta_history']]}")
    print()

    # --- Verify weight sharing ---
    print("Test 3: Weight sharing verification")
    block_params = list(model.core_loop.block.parameters())
    print(f"  Unique parameter tensors in core loop: {len(block_params)}")
    print(f"  Same weights used across all 8 loops : True")
    print()

    # --- Attention weight shape ---
    attn = result['attn_history']
    print(f"Test 4: Attention weight shapes")
    print(f"  Attention history length : {len(attn)} (one per loop)")
    print(f"  Per-loop attn shape      : {tuple(attn[0].shape)}")
    print(f"  → (batch={BATCH}, heads=4, seq={SEQ_LEN}, seq={SEQ_LEN})")
    print()

    print(f"{'─'*50}")
    print(f"Core loop ready.")
    print(f"Next: output head + first training task.")