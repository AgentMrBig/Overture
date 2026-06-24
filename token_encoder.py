"""
Omnicapable Transformer — Token Encoding Layer
==============================================
Converts any domain input into a complex-valued sparse token
in the shared latent space.

Token format: complex vector of shape (seq_len, d_model)
  - Complex: each dimension has magnitude + phase
  - Sparse:  only top-k dimensions active per token
  - Shared:  same format regardless of input domain

Usage:
    encoder = TokenEncoder(input_dim=12, d_model=64, k_sparse=16)
    tokens  = encoder(x)   # x: (batch, seq_len, input_dim) real
                           # out: (batch, seq_len, d_model) complex
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
#  1. DOMAIN PROJECTION
#     Lightweight linear layer that maps domain-specific
#     raw input into a common real-valued space before
#     complex lifting. One per input domain, all project
#     to the same d_model*2 intermediate dimension.
# ─────────────────────────────────────────────────────────

class DomainProjection(nn.Module):
    """
    Maps raw domain input → intermediate real space.

    Args:
        input_dim : dimensionality of raw input features
        d_model   : target model dimension (shared across domains)
    """
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        # Project to 2*d_model — we'll split into real/imag halves
        self.proj = nn.Linear(input_dim, d_model * 2)
        self.norm = nn.LayerNorm(d_model * 2)

        # Initialize with small weights — don't dominate the complex lift
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:   (batch, seq_len, input_dim)
        out: (batch, seq_len, d_model * 2)
        """
        return self.norm(self.proj(x))


# ─────────────────────────────────────────────────────────
#  2. COMPLEX LIFTING
#     Converts a real-valued vector into a complex vector.
#     Two approaches available:
#
#     A) Split — first half becomes real, second becomes imag
#        Simple, direct, no information loss
#
#     B) Polar — learn magnitude and phase explicitly
#        More expressive, magnitude >= 0 enforced by softplus,
#        phase in [-π, π] enforced by tanh*π
#        Better for temporal/cyclical data like price or audio
# ─────────────────────────────────────────────────────────

class ComplexLift(nn.Module):
    """
    Lifts a real vector of size (d_model*2) into a complex
    vector of size (d_model).

    Args:
        d_model : complex output dimension
        mode    : 'split' or 'polar'
    """
    def __init__(self, d_model: int, mode: str = 'polar'):
        super().__init__()
        assert mode in ('split', 'polar'), "mode must be 'split' or 'polar'"
        self.d_model = d_model
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:   (batch, seq_len, d_model * 2) real
        out: (batch, seq_len, d_model)     complex
        """
        # Split input into two halves
        a, b = x.chunk(2, dim=-1)   # each: (batch, seq_len, d_model)

        if self.mode == 'split':
            # Direct: a is real part, b is imaginary part
            return torch.complex(a, b)

        elif self.mode == 'polar':
            # Polar: a → magnitude (must be >= 0), b → phase (in [-π, π])
            magnitude = F.softplus(a)           # smooth, always positive
            phase     = torch.tanh(b) * torch.pi  # bounded [-π, π]

            real = magnitude * torch.cos(phase)
            imag = magnitude * torch.sin(phase)
            return torch.complex(real, imag)


# ─────────────────────────────────────────────────────────
#  3. LEARNED SPARSITY MASK
#     For each token, selects the top-k most important
#     dimensions and zeros out the rest.
#
#     The "importance" of each dimension is learned —
#     a small network scores each dimension and only
#     the top-k survive. This forces specialization:
#     different input types activate different subspaces.
#
#     We operate on the magnitude of the complex token
#     to decide which dimensions to keep, then apply
#     the mask to both real and imaginary parts.
# ─────────────────────────────────────────────────────────

class SparsityMask(nn.Module):
    """
    Applies learned top-k sparsity to complex tokens.

    Args:
        d_model  : complex token dimension
        k_sparse : number of active dimensions per token
        learnable: if True, learn a scoring network for importance
                   if False, use raw magnitude (simpler baseline)
    """
    def __init__(self, d_model: int, k_sparse: int, learnable: bool = True):
        super().__init__()
        assert k_sparse <= d_model, "k_sparse must be <= d_model"
        self.d_model   = d_model
        self.k_sparse  = k_sparse
        self.learnable = learnable

        if learnable:
            # Small MLP that scores each dimension's importance
            # Input: magnitude of each complex dimension
            # Output: importance score per dimension
            self.scorer = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z:   (batch, seq_len, d_model) complex
        out: (batch, seq_len, d_model) complex, sparse
        """
        # Compute magnitude of each complex dimension
        magnitude = z.abs()   # (batch, seq_len, d_model) real

        if self.learnable:
            # Score dimensions based on learned importance
            scores = self.scorer(magnitude)   # (batch, seq_len, d_model)
        else:
            # Baseline: use raw magnitude as score
            scores = magnitude

        # Find top-k dimension indices per token
        _, top_k_indices = torch.topk(scores, self.k_sparse, dim=-1)

        # Build binary mask — 1 for active dims, 0 for inactive
        mask = torch.zeros_like(magnitude)
        mask.scatter_(-1, top_k_indices, 1.0)

        # Apply mask to both real and imaginary parts
        real_masked = z.real * mask
        imag_masked = z.imag * mask

        return torch.complex(real_masked, imag_masked)


# ─────────────────────────────────────────────────────────
#  4. POSITIONAL ENCODING (Complex)
#     Standard sinusoidal positional encoding but lifted
#     into complex space — the encoding itself becomes
#     a complex rotation, which is the natural way to
#     encode position for a complex-valued sequence.
#
#     pos_encoding[pos, 2i]   = cos(pos / 10000^(2i/d))
#     pos_encoding[pos, 2i+1] = sin(pos / 10000^(2i/d))
#
#     Lifted to complex: e^(i * pos / 10000^(2i/d))
#     This is RoPE-style (Rotary Position Embedding) —
#     position encoded as a rotation in complex space.
# ─────────────────────────────────────────────────────────

class ComplexPositionalEncoding(nn.Module):
    """
    Encodes sequence position as complex rotations.
    Adds positional information without extra parameters.

    Args:
        d_model  : complex token dimension
        max_len  : maximum sequence length
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model

        # Precompute rotation frequencies
        position = torch.arange(max_len).unsqueeze(1).float()
        freq     = torch.pow(
            10000.0,
            -torch.arange(0, d_model).float() / d_model
        )
        # Phase angle per position per dimension
        angles = position * freq   # (max_len, d_model)

        # Store as complex rotors: e^(i*angle) = cos(angle) + i*sin(angle)
        rotors = torch.complex(torch.cos(angles), torch.sin(angles))
        self.register_buffer('rotors', rotors)   # not a parameter, just cached

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z:   (batch, seq_len, d_model) complex
        out: (batch, seq_len, d_model) complex — rotated by position
        """
        seq_len = z.size(1)
        # Multiply each token by its positional rotor
        # Complex multiplication = rotation in the complex plane
        return z * self.rotors[:seq_len].unsqueeze(0)


# ─────────────────────────────────────────────────────────
#  5. FULL TOKEN ENCODER
#     Assembles all components into a single module.
#     One TokenEncoder per input domain.
#     All encoders share the same d_model and k_sparse.
# ─────────────────────────────────────────────────────────

class TokenEncoder(nn.Module):
    """
    Full token encoding pipeline for one input domain.

    Converts raw domain input → complex sparse token
    in the shared latent space.

    Args:
        input_dim   : dimensionality of raw input (e.g. 12 for OHLCV)
        d_model     : shared complex token dimension
        k_sparse    : active dimensions per token
        lift_mode   : 'polar' or 'split' for complex lifting
        max_seq_len : maximum sequence length for positional encoding
    """
    def __init__(
        self,
        input_dim   : int,
        d_model     : int  = 64,
        k_sparse    : int  = 16,
        lift_mode   : str  = 'polar',
        max_seq_len : int  = 512,
    ):
        super().__init__()
        self.d_model  = d_model
        self.k_sparse = k_sparse

        self.domain_proj = DomainProjection(input_dim, d_model)
        self.complex_lift = ComplexLift(d_model, mode=lift_mode)
        self.sparsity    = SparsityMask(d_model, k_sparse, learnable=True)
        self.pos_enc     = ComplexPositionalEncoding(d_model, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:   (batch, seq_len, input_dim)  real   ← raw domain input
        out: (batch, seq_len, d_model)    complex ← shared latent token
        """
        # 1. Project to shared real space
        projected = self.domain_proj(x)           # (B, S, d_model*2) real

        # 2. Lift to complex
        complex_token = self.complex_lift(projected)  # (B, S, d_model) complex

        # 3. Apply positional encoding (complex rotation)
        positioned = self.pos_enc(complex_token)      # (B, S, d_model) complex

        # 4. Apply learned sparsity
        sparse_token = self.sparsity(positioned)      # (B, S, d_model) complex sparse

        return sparse_token

    def token_stats(self, tokens: torch.Tensor) -> dict:
        """
        Diagnostic utility — returns stats about a batch of tokens.
        Useful for verifying sparsity and magnitude distribution.
        """
        magnitude  = tokens.abs()
        active     = (magnitude > 1e-6).float()
        sparsity   = active.mean().item()

        return {
            'mean_magnitude'   : magnitude.mean().item(),
            'max_magnitude'    : magnitude.max().item(),
            'active_fraction'  : sparsity,
            'active_per_token' : active.sum(dim=-1).mean().item(),
            'phase_std'        : tokens.angle().std().item(),
        }


# ─────────────────────────────────────────────────────────
#  6. MULTI-DOMAIN REGISTRY
#     Manages multiple domain encoders that all map
#     into the same shared latent space.
#     Add new domains without touching the core loop.
# ─────────────────────────────────────────────────────────

class DomainRegistry(nn.Module):
    """
    Registry of domain-specific TokenEncoders.
    All encoders share d_model and k_sparse.
    New domains can be registered at any time.

    Example:
        registry = DomainRegistry(d_model=64, k_sparse=16)
        registry.register('price',    input_dim=12)
        registry.register('language', input_dim=384)
        registry.register('audio',    input_dim=80)

        price_tokens    = registry('price',    price_data)
        language_tokens = registry('language', text_embeddings)
    """
    def __init__(self, d_model: int = 64, k_sparse: int = 16):
        super().__init__()
        self.d_model  = d_model
        self.k_sparse = k_sparse
        self.encoders = nn.ModuleDict()

    def register(self, domain_name: str, input_dim: int, lift_mode: str = 'polar'):
        """Register a new domain encoder."""
        if domain_name in self.encoders:
            print(f"[Registry] Warning: overwriting existing encoder '{domain_name}'")
        self.encoders[domain_name] = TokenEncoder(
            input_dim   = input_dim,
            d_model     = self.d_model,
            k_sparse    = self.k_sparse,
            lift_mode   = lift_mode,
        )
        print(f"[Registry] Registered domain '{domain_name}' "
              f"(input_dim={input_dim}, d_model={self.d_model}, k={self.k_sparse})")

    def forward(self, domain_name: str, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input from a named domain.
        x:   (batch, seq_len, input_dim) — domain's native format
        out: (batch, seq_len, d_model)   — shared complex sparse token
        """
        if domain_name not in self.encoders:
            raise KeyError(f"Domain '{domain_name}' not registered. "
                           f"Available: {list(self.encoders.keys())}")
        return self.encoders[domain_name](x)

    def list_domains(self):
        return list(self.encoders.keys())


# ─────────────────────────────────────────────────────────
#  TEST / SMOKE CHECK
#  Run this file directly to verify everything works
#  on your RTX machine:
#      python token_encoder.py
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # --- Config ---
    BATCH    = 4
    SEQ_LEN  = 60    # 60 candles / 60 words / 60 frames
    D_MODEL  = 64    # shared complex token dimension
    K_SPARSE = 16    # 16 of 64 dims active per token (25% density)

    # --- Build registry ---
    registry = DomainRegistry(d_model=D_MODEL, k_sparse=K_SPARSE)
    registry.register('price',    input_dim=12)   # OHLCV + derived
    registry.register('language', input_dim=384)  # sentence embedding dim
    registry.register('audio',    input_dim=80)   # mel spectrogram bins
    registry.to(device)

    print(f"{'─'*50}")
    print(f"Token shape: ({BATCH}, {SEQ_LEN}, {D_MODEL}) complex")
    print(f"Sparsity:    {K_SPARSE}/{D_MODEL} active dims = {K_SPARSE/D_MODEL*100:.0f}% density")
    print(f"{'─'*50}\n")

    # --- Test each domain ---
    domains = {
        'price'   : torch.randn(BATCH, SEQ_LEN, 12),
        'language': torch.randn(BATCH, SEQ_LEN, 384),
        'audio'   : torch.randn(BATCH, SEQ_LEN, 80),
    }

    for domain, raw_input in domains.items():
        raw_input = raw_input.to(device)

        with torch.no_grad():
            tokens = registry(domain, raw_input)

        encoder = registry.encoders[domain]
        stats   = encoder.token_stats(tokens)

        print(f"Domain: {domain}")
        print(f"  Input shape  : {tuple(raw_input.shape)}")
        print(f"  Token shape  : {tuple(tokens.shape)}  dtype={tokens.dtype}")
        print(f"  Active dims  : {stats['active_per_token']:.1f} / {D_MODEL}")
        print(f"  Mean magnitude: {stats['mean_magnitude']:.4f}")
        print(f"  Phase std     : {stats['phase_std']:.4f} rad")
        print()

    # --- Parameter count ---
    total_params = sum(p.numel() for p in registry.parameters())
    print(f"{'─'*50}")
    print(f"Total encoder parameters: {total_params:,}")
    print(f"Per domain avg:           {total_params // len(domains):,}")
    print(f"{'─'*50}")

    # --- Verify tokens from different domains have same shape ---
    price_tok = registry('price',    torch.randn(BATCH, SEQ_LEN, 12).to(device))
    lang_tok  = registry('language', torch.randn(BATCH, SEQ_LEN, 384).to(device))

    assert price_tok.shape == lang_tok.shape, "Shape mismatch between domains!"
    assert price_tok.dtype == torch.complex64, f"Expected complex64, got {price_tok.dtype}"
    print(f"\nAll assertions passed.")
    print(f"Price tokens and language tokens share shape: {tuple(price_tok.shape)}")
    print(f"\nToken encoder layer ready. Next: core loop / attention mechanism.")
