"""
Overture REPL
=============
Interactive terminal interface for the Overture frame.
Type 'help' for commands.

Usage:
    python overture.py
"""

import torch
import numpy as np
import time
import sys
import os

# ─────────────────────────────────────────────────────────
#  TERMINAL COLORS
# ─────────────────────────────────────────────────────────

class C:
    RESET  = '\033[0m'
    BOLD   = '\033[1m'
    DIM    = '\033[2m'
    CYAN   = '\033[96m'
    BLUE   = '\033[94m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    PURPLE = '\033[95m'
    WHITE  = '\033[97m'
    GRAY   = '\033[90m'

def cyan(s):   return f"{C.CYAN}{s}{C.RESET}"
def blue(s):   return f"{C.BLUE}{s}{C.RESET}"
def green(s):  return f"{C.GREEN}{s}{C.RESET}"
def yellow(s): return f"{C.YELLOW}{s}{C.RESET}"
def red(s):    return f"{C.RED}{s}{C.RESET}"
def purple(s): return f"{C.PURPLE}{s}{C.RESET}"
def gray(s):   return f"{C.GRAY}{s}{C.RESET}"
def bold(s):   return f"{C.BOLD}{s}{C.RESET}"
def dim(s):    return f"{C.DIM}{s}{C.RESET}"


# ─────────────────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────────────────

BANNER = f"""
{cyan('╔══════════════════════════════════════════════════════╗')}
{cyan('║')}  {bold(cyan('OVERTURE'))}  {gray('v0.1')}                                      {cyan('║')}
{cyan('║')}  {dim('omnicapable transformer frame')}                      {cyan('║')}
{cyan('║')}  {dim('complex · sparse · iterative · domain-agnostic')}     {cyan('║')}
{cyan('╚══════════════════════════════════════════════════════╝')}
"""

HELP_TEXT = f"""
{bold(cyan('Commands:'))}

  {cyan('encode')} {yellow('"text"')}              encode a sentence → show token stats
  {cyan('encode')} {yellow('path/to/image.jpg')}   encode an image   → show token stats
  {cyan('similarity')} {yellow('"text a" "text b"')} similarity in complex space
  {cyan('compare')} {yellow('"a" "b" "c"')}        compare multiple inputs pairwise
  {cyan('cluster')} {yellow('word1 word2 ...')}     group concepts by proximity
  {cyan('search')} {yellow('"query"')} {yellow('"c1" "c2" ...')} find closest candidate to query
  {cyan('loops')} {yellow('"text"')}               show loop-by-loop convergence
  {cyan('status')}                      model config and parameter count
  {cyan('help')}                        show this message
  {cyan('exit')}                        quit

{bold(cyan('Notes:'))}
  {gray('- Weights are randomly initialized until trained')}
  {gray('- Similarity scores reflect Qwen3-VL semantic structure')}
  {gray('- Token stats show complex space representation')}
  {gray('- Loop convergence shows iterative refinement in action')}
"""


# ─────────────────────────────────────────────────────────
#  FRAME — wraps the model stack
# ─────────────────────────────────────────────────────────

class OvertureFrame:
    """
    Wraps Qwen3-VL + our domain encoder + core loop.
    Provides clean encode/similarity/cluster methods for the REPL.
    """

    def __init__(self, device, d_model=64, k_sparse=16, truncate_dim=64):
        self.device      = device
        self.d_model     = d_model
        self.k_sparse    = k_sparse
        self.truncate_dim= truncate_dim
        self.qwen        = None
        self.registry    = None
        self.core        = None

    def load(self):
        from sentence_transformers import SentenceTransformer
        from token_encoder import DomainRegistry
        from core_loop import CoreLoop

        print(f"  {dim('Loading Qwen3-VL-Embedding-2B...')}", end='', flush=True)
        t0 = time.perf_counter()
        self.qwen = SentenceTransformer(
            "Qwen/Qwen3-VL-Embedding-2B",
            trust_remote_code=True,
            truncate_dim=self.truncate_dim,
        )
        self.qwen.to(self.device)
        self.qwen_time = time.perf_counter() - t0
        print(f"\r  {green('Qwen3-VL loaded')} {gray(f'({self.qwen_time:.1f}s, truncate_dim={self.truncate_dim})')}")

        print(f"  {dim('Building frame...')}", end='', flush=True)
        self.registry = DomainRegistry(d_model=self.d_model, k_sparse=self.k_sparse)
        self.registry.register('qwen', input_dim=self.truncate_dim)
        self.registry.to(self.device).eval()

        self.core = CoreLoop(
            d_model       = self.d_model,
            n_heads       = 4,
            ff_multiplier = 4,
            max_loops     = 8,
        )
        self.core.to(self.device).eval()

        total_params = (
            sum(p.numel() for p in self.registry.parameters()) +
            sum(p.numel() for p in self.core.parameters())
        )
        print(f"\r  {green('Frame ready')} {gray(f'({total_params:,} params)')}")

    def _qwen_encode(self, inputs):
        """Encode text or image paths with Qwen3-VL."""
        with torch.no_grad():
            embs = self.qwen.encode(
                inputs,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if not isinstance(embs, torch.Tensor):
                embs = torch.tensor(embs)
            return embs.to(self.device).float()

    def _to_complex_tokens(self, embs):
        """Push Qwen embeddings through our domain encoder."""
        with torch.no_grad():
            embs_3d = embs.unsqueeze(1)                    # (N, 1, truncate_dim)
            tokens  = self.registry('qwen', embs_3d)       # (N, 1, d_model) complex
            return tokens.squeeze(1)                        # (N, d_model) complex

    def _run_loop(self, tokens, return_history=False):
        """Run the core loop on a sequence of tokens."""
        with torch.no_grad():
            tokens_seq = tokens.unsqueeze(0)               # (1, N, d_model)
            result = self.core(tokens_seq, return_history=return_history)
            return result

    def encode_single(self, text_or_path):
        """Encode one text or image. Returns (qwen_emb, complex_token, loop_result)."""
        t0 = time.perf_counter()
        emb   = self._qwen_encode([text_or_path])          # (1, truncate_dim)
        token = self._to_complex_tokens(emb)               # (1, d_model) complex
        loop  = self._run_loop(token, return_history=True)
        elapsed = time.perf_counter() - t0
        return emb[0], token[0], loop, elapsed

    def encode_batch(self, inputs):
        """Encode multiple texts/images. Returns (embs, tokens)."""
        embs   = self._qwen_encode(inputs)                 # (N, truncate_dim)
        tokens = self._to_complex_tokens(embs)             # (N, d_model) complex
        return embs, tokens

    def complex_sim(self, a, b):
        """Cosine similarity in complex token space (magnitude-weighted)."""
        mag_a = a.abs().float()
        mag_b = b.abs().float()
        dot   = (mag_a * mag_b).sum()
        norm  = mag_a.norm().clamp(min=1e-8) * mag_b.norm().clamp(min=1e-8)
        return (dot / norm).item()

    def qwen_sim(self, a, b):
        """Cosine similarity in Qwen embedding space."""
        a = a.float().flatten()
        b = b.float().flatten()
        return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


# ─────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────────────────

def token_bar(magnitude, width=32):
    """Visual bar showing token magnitude."""
    filled = int(magnitude * width * 3)
    filled = min(filled, width)
    bar = '█' * filled + '░' * (width - filled)
    return bar

def phase_indicator(phase_std):
    """Describe phase spread."""
    if phase_std > 2.5:   return green('wide  (full phase range)')
    elif phase_std > 1.5: return cyan('medium')
    else:                 return yellow('narrow (compressed phase)')

def sim_bar(sim, width=20):
    """Visual similarity bar."""
    filled = int(sim * width)
    filled = max(0, min(filled, width))
    if sim > 0.75:   color = green
    elif sim > 0.45: color = cyan
    elif sim > 0.25: color = yellow
    else:            color = gray
    bar = '█' * filled + '░' * (width - filled)
    return color(bar) + f' {sim:.4f}'

def parse_quoted_args(arg_string):
    """Parse a mix of quoted strings and bare words from a command."""
    import re
    tokens = []
    # Find all quoted strings
    quoted = re.findall(r'"([^"]*)"', arg_string)
    # Find remaining bare words (not inside quotes)
    remainder = re.sub(r'"[^"]*"', '', arg_string).split()
    return quoted + remainder


# ─────────────────────────────────────────────────────────
#  COMMAND HANDLERS
# ─────────────────────────────────────────────────────────

def cmd_encode(frame, args):
    if not args:
        print(red('  Usage: encode "text" or encode path/to/image.jpg'))
        return

    # Detect if it's a file path or text
    input_str = args[0] if args else ''
    is_image  = os.path.isfile(input_str) and input_str.lower().endswith(
        ('.jpg', '.jpeg', '.png', '.webp', '.gif')
    )
    input_type = 'image' if is_image else 'text'

    print(f"\n  {dim('Encoding')} {yellow(repr(input_str))} {dim(f'as {input_type}...')}")
    t0 = time.perf_counter()

    qwen_emb, token, loop_result, elapsed = frame.encode_single(input_str)

    mag        = token.abs().mean().item()
    mag_max    = token.abs().max().item()
    phase_std  = token.angle().std().item()
    active     = (token.abs() > 1e-6).sum().item()
    loops_run  = loop_result['loops_run']
    converged  = loop_result['converged']

    print(f"\n  {bold('Qwen3-VL embedding')} {gray(f'({frame.truncate_dim}-dim real)')}")
    print(f"  Norm      : {qwen_emb.norm().item():.4f}")

    print(f"\n  {bold('Complex token')} {gray(f'(d_model={frame.d_model})')}")
    print(f"  Magnitude : {token_bar(mag)} {mag:.4f} mean  {mag_max:.4f} max")
    print(f"  Phase     : {phase_indicator(phase_std)} {gray(f'std={phase_std:.3f}r')}")
    print(f"  Active    : {green(str(int(active)))}{gray(f'/{frame.d_model}')} dims  {gray(f'({active/frame.d_model:.0%} density)')}")

    print(f"\n  {bold('Core loop')}")
    print(f"  Iterations: {cyan(str(loops_run))}{gray(f'/{frame.core.max_loops}')}")
    converged_str = green('yes (early exit)') if converged else yellow('no (ran to max)')
    print(f"  Converged : {converged_str}")

    if loop_result['delta_history']:
        deltas = loop_result['delta_history']
        delta_str = '  '.join([f"{d:.3f}" for d in deltas])
        print(f"  Deltas    : {gray(delta_str)}")

    print(f"\n  {gray(f'Total: {elapsed:.3f}s')}")


def cmd_similarity(frame, args):
    inputs = parse_quoted_args(' '.join(args))
    if len(inputs) < 2:
        print(red('  Usage: similarity "text a" "text b"'))
        return

    a_str, b_str = inputs[0], inputs[1]
    print(f"\n  {dim('Computing similarity...')}")

    embs, tokens = frame.encode_batch([a_str, b_str])

    qwen_s    = frame.qwen_sim(embs[0], embs[1])
    complex_s = frame.complex_sim(tokens[0], tokens[1])

    print(f"\n  {bold('A:')} {yellow(repr(a_str[:60]))}")
    print(f"  {bold('B:')} {yellow(repr(b_str[:60]))}")
    print()
    print(f"  Qwen space    : {sim_bar(qwen_s)}")
    print(f"  Complex space : {sim_bar(complex_s)}")
    print()

    if complex_s > 0.75:
        interp = green('Very similar — closely related concepts')
    elif complex_s > 0.55:
        interp = cyan('Moderately similar — related domain or theme')
    elif complex_s > 0.35:
        interp = yellow('Weakly similar — some shared context')
    else:
        interp = gray('Dissimilar — different domains or meanings')
    print(f"  {interp}")


def cmd_compare(frame, args):
    inputs = parse_quoted_args(' '.join(args))
    if len(inputs) < 2:
        print(red('  Usage: compare "a" "b" "c" ...'))
        return

    print(f"\n  {dim(f'Comparing {len(inputs)} inputs...')}")
    embs, tokens = frame.encode_batch(inputs)

    n = len(inputs)
    labels = [repr(s[:20]) for s in inputs]

    # Header
    print()
    print(f"  {'':22}", end='')
    for lbl in labels:
        print(f"  {cyan(lbl):>24}", end='')
    print()
    print(f"  {'─'*(24 + 24*n)}")

    # Matrix
    for i in range(n):
        print(f"  {yellow(labels[i]):<22}", end='')
        for j in range(n):
            sim = frame.complex_sim(tokens[i], tokens[j])
            if i == j:
                print(f"  {gray('1.0000'):>22}", end='')
            elif sim > 0.65:
                print(f"  {green(f'{sim:.4f}'):>22}", end='')
            elif sim > 0.40:
                print(f"  {cyan(f'{sim:.4f}'):>22}", end='')
            else:
                print(f"  {gray(f'{sim:.4f}'):>22}", end='')
        print()


def cmd_cluster(frame, args):
    inputs = parse_quoted_args(' '.join(args))
    if len(inputs) < 3:
        print(red('  Usage: cluster word1 word2 word3 ...'))
        return

    print(f"\n  {dim(f'Clustering {len(inputs)} concepts...')}")
    embs, tokens = frame.encode_batch(inputs)

    n = len(inputs)

    # Find nearest neighbor for each
    print(f"\n  {bold('Nearest neighbors in complex space:')}\n")
    clusters = {}
    for i in range(n):
        best_sim  = -1
        best_j    = -1
        for j in range(n):
            if i == j: continue
            s = frame.complex_sim(tokens[i], tokens[j])
            if s > best_sim:
                best_sim = s
                best_j   = j
        nn_label = inputs[best_j]
        sim_str  = sim_bar(best_sim, width=12)
        li = f"{inputs[i]:<20}"; nl = f"{nn_label:<20}"
        print(f"  {yellow(li)} -> {cyan(nl)} {sim_str}")
        if nn_label not in clusters:
            clusters[nn_label] = []
        clusters[nn_label].append(inputs[i])

    # Show emergent groups
    print(f"\n  {bold('Emergent groups:')}\n")
    seen = set()
    group_num = 1
    for i in range(n):
        if inputs[i] in seen:
            continue
        group = [inputs[i]]
        seen.add(inputs[i])
        for j in range(n):
            if i == j or inputs[j] in seen:
                continue
            if frame.complex_sim(tokens[i], tokens[j]) > 0.55:
                group.append(inputs[j])
                seen.add(inputs[j])
        if len(group) > 1:
            items = '  '.join([cyan(g) for g in group])
            print(f"  Group {group_num}: {items}")
            group_num += 1

    # Singletons
    singletons = [inputs[i] for i in range(n) if inputs[i] not in seen]
    if singletons:
        items = '  '.join([gray(s) for s in singletons])
        print(f"  Standalone: {items}")


def cmd_search(frame, args):
    inputs = parse_quoted_args(' '.join(args))
    if len(inputs) < 2:
        print(red('  Usage: search "query" "candidate1" "candidate2" ...'))
        return

    query      = inputs[0]
    candidates = inputs[1:]

    print(f"\n  {dim('Searching...')}")
    all_inputs    = [query] + candidates
    embs, tokens  = frame.encode_batch(all_inputs)
    query_token   = tokens[0]
    cand_tokens   = tokens[1:]

    print(f"\n  {bold('Query:')} {yellow(repr(query[:60]))}")
    print(f"\n  {bold('Candidates ranked by similarity:')}\n")

    ranked = []
    for i, (cand, ct) in enumerate(zip(candidates, cand_tokens)):
        sim = frame.complex_sim(query_token, ct)
        ranked.append((sim, cand))
    ranked.sort(reverse=True)

    for rank, (sim, cand) in enumerate(ranked, 1):
        marker = green('→') if rank == 1 else gray(' ')
        print(f"  {marker} {rank}. {yellow(repr(cand[:40])):<44} {sim_bar(sim, width=16)}")


def cmd_loops(frame, args):
    inputs = parse_quoted_args(' '.join(args))
    if not inputs:
        print(red('  Usage: loops "text"'))
        return

    text = inputs[0]
    print(f"\n  {dim('Running loop analysis...')}")

    emb   = frame._qwen_encode([text])
    token = frame._to_complex_tokens(emb)

    with torch.no_grad():
        token_seq = token.unsqueeze(0)
        result    = frame.core(token_seq, return_history=True)

    print(f"\n  {bold('Input:')} {yellow(repr(text[:60]))}")
    print(f"\n  {bold('Representation evolution across loops:')}\n")
    print(f"  {'Loop':>6}  {'Delta':>8}  {'Magnitude':>10}  {'Stability'}")
    print(f"  {'─'*50}")

    history = result['repr_history']
    deltas  = result['delta_history']

    for i, h in enumerate(history[1:], 1):
        rep   = h[0, 0]
        mag   = rep.abs().mean().item()
        delta = deltas[i-1] if i-1 < len(deltas) else 0

        if delta < 0.05:   stability = green('converging  ██████')
        elif delta < 0.15: stability = cyan('settling    ████░░')
        elif delta < 0.30: stability = yellow('refining    ██░░░░')
        else:              stability = gray('exploring   █░░░░░')

        print(f"  {i:>6}  {delta:>8.4f}  {mag:>10.4f}  {stability}")

    loops_run = result['loops_run']
    converged = result['converged']
    print(f"\n  Loops run : {cyan(str(loops_run))}/{frame.core.max_loops}")
    print(f"  Converged : {green('yes') if converged else yellow('not yet (untrained weights)')}")


def cmd_status(frame):
    encoder_params = sum(p.numel() for p in frame.registry.parameters())
    loop_params    = sum(p.numel() for p in frame.core.parameters())
    total          = encoder_params + loop_params

    device_str = str(frame.device)
    if frame.device.type == 'cuda':
        device_str = f"cuda ({torch.cuda.get_device_name(0)})"

    print(f"\n  {bold(cyan('OVERTURE STATUS'))}\n")
    print(f"  {bold('Hardware')}")
    print(f"    Device        : {green(device_str)}")
    print(f"    VRAM used     : {torch.cuda.memory_allocated()/1e6:.0f} MB" if frame.device.type == 'cuda' else '')

    print(f"\n  {bold('Domain Parts')}")
    print(f"    Qwen3-VL-2B   : {green('loaded')}  {gray('(frozen, 2B params)')}")
    print(f"    truncate_dim  : {cyan(str(frame.truncate_dim))}")
    print(f"    Audio (Whisper): {yellow('not loaded')}")
    print(f"    Price encoder : {yellow('not loaded')}")

    print(f"\n  {bold('Frame Architecture')}")
    print(f"    d_model       : {cyan(str(frame.d_model))}")
    print(f"    k_sparse      : {cyan(str(frame.k_sparse))}  {gray(f'({frame.k_sparse/frame.d_model:.0%} density)')}")
    print(f"    n_heads       : {cyan('4')}")
    print(f"    max_loops     : {cyan(str(frame.core.max_loops))}")
    print(f"    ff_dim        : {cyan(str(frame.d_model * 4))}")

    print(f"\n  {bold('Parameters')}")
    print(f"    Domain encoder: {cyan(f'{encoder_params:,}')}")
    print(f"    Core loop     : {cyan(f'{loop_params:,}')}  {gray('(shared across all loops)')}")
    print(f"    Total         : {cyan(f'{total:,}')}")
    print(f"    Qwen3-VL      : {gray('~2,000,000,000  (frozen, not counted)')}")

    print(f"\n  {bold('Training state')}")
    print(f"    Core loop     : {yellow('randomly initialized')}")
    print(f"    Lang encoder  : {yellow('randomly initialized')}")
    print(f"    Checkpoint    : {green('overture_best.pt') if os.path.exists('overture_best.pt') else gray('none')}")


# ─────────────────────────────────────────────────────────
#  MAIN REPL LOOP
# ─────────────────────────────────────────────────────────

def main():
    print(BANNER)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  {dim('Device:')} {green(str(device))}")
    if device.type == 'cuda':
        print(f"  {dim('GPU   :')} {green(torch.cuda.get_device_name(0))}")
    print()

    frame = OvertureFrame(device)

    print(f"  {bold('Initializing frame...')}")
    try:
        frame.load()
    except Exception as e:
        print(red(f"\n  Failed to initialize: {e}"))
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n  {green('Frame ready.')} Type {cyan('help')} for commands.\n")

    # ── REPL ──
    while True:
        try:
            raw = input(f"{cyan('>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {gray('Goodbye.')}")
            break

        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd   = parts[0].lower()
        args  = parts[1].split() if len(parts) > 1 else []
        rest  = parts[1] if len(parts) > 1 else ''

        try:
            if cmd in ('exit', 'quit', 'q'):
                print(f"\n  {gray('Goodbye.')}")
                break

            elif cmd == 'help':
                print(HELP_TEXT)

            elif cmd == 'status':
                cmd_status(frame)

            elif cmd == 'encode':
                inputs = parse_quoted_args(rest)
                cmd_encode(frame, inputs)

            elif cmd == 'similarity' or cmd == 'sim':
                cmd_similarity(frame, [rest])

            elif cmd == 'compare':
                cmd_compare(frame, [rest])

            elif cmd == 'cluster':
                cmd_cluster(frame, [rest])

            elif cmd == 'search':
                cmd_search(frame, [rest])

            elif cmd == 'loops':
                cmd_loops(frame, [rest])

            else:
                print(gray(f"  Unknown command: '{cmd}'. Type 'help' for commands."))

        except Exception as e:
            print(red(f"\n  Error: {e}"))
            import traceback
            traceback.print_exc()

        print()


if __name__ == '__main__':
    main()