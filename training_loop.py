"""
Omnicapable Transformer — Training Loop
========================================
Trains the full pipeline (encoder → core loop → output head)
on a synthetic dataset designed to verify that:

    1. Gradients flow correctly through complex layers
    2. The loss drops — the network is actually learning
    3. Loop count adapts — fewer loops needed as confidence grows
    4. The pattern generalizes — test loss tracks train loss

Synthetic Task: Sequence Pattern Classification
    Input  : (batch, seq_len, 12) — fake price-like features
    Pattern: if the mean of the first 10 timesteps > mean of last 10,
             label = 1 (uptrend), else label = 0 (downtrend)
    This is a learnable temporal pattern that requires the network
    to compare early vs late parts of the sequence — a real
    attention challenge, not just a trivial lookup.

Usage:
    python training_loop.py

After training, the script saves a checkpoint and runs evaluation
showing per-sample predictions vs ground truth.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
from output_head import OvertureWithHead


# ─────────────────────────────────────────────────────────
#  1. SYNTHETIC DATASET
#     Generates fake price-like sequences with a planted
#     temporal pattern that the network must learn to detect.
# ─────────────────────────────────────────────────────────

class SyntheticSequenceDataset(Dataset):
    """
    Synthetic dataset for verifying network learning.

    Each sample is a (seq_len, feature_dim) tensor.
    Label is determined by a planted temporal pattern.

    Pattern options:
        'trend'    — early mean vs late mean comparison
        'spike'    — is there a spike in the middle third?
        'momentum' — is the sequence monotonically rising?

    Args:
        n_samples   : number of samples to generate
        seq_len     : sequence length
        feature_dim : number of features per timestep
        pattern     : which pattern to plant
        noise       : noise level added on top of pattern
        seed        : random seed for reproducibility
    """
    def __init__(
        self,
        n_samples   : int   = 2000,
        seq_len     : int   = 60,
        feature_dim : int   = 12,
        pattern     : str   = 'trend',
        noise       : float = 0.5,
        seed        : int   = 42,
    ):
        super().__init__()
        self.n_samples   = n_samples
        self.seq_len     = seq_len
        self.feature_dim = feature_dim
        self.pattern     = pattern

        rng = np.random.RandomState(seed)

        # Generate base sequences — random noise
        data = rng.randn(n_samples, seq_len, feature_dim).astype(np.float32)

        # Plant pattern in first feature dimension
        labels = np.zeros(n_samples, dtype=np.float32)

        if pattern == 'trend':
            # Uptrend: inject rising signal into first feature
            # Downtrend: inject falling signal
            for i in range(n_samples):
                if rng.rand() > 0.5:
                    # Uptrend — add rising ramp to first feature
                    ramp = np.linspace(-1, 1, seq_len)
                    data[i, :, 0] += ramp * 2.0
                    labels[i] = 1.0
                else:
                    # Downtrend — add falling ramp
                    ramp = np.linspace(1, -1, seq_len)
                    data[i, :, 0] += ramp * 2.0
                    labels[i] = 0.0

        elif pattern == 'spike':
            # Spike in middle third → label 1, no spike → label 0
            mid_start = seq_len // 3
            mid_end   = 2 * seq_len // 3
            for i in range(n_samples):
                if rng.rand() > 0.5:
                    spike_pos = rng.randint(mid_start, mid_end)
                    data[i, spike_pos, 0] += 5.0
                    labels[i] = 1.0

        elif pattern == 'momentum':
            # Monotonically rising last 20 steps → label 1
            for i in range(n_samples):
                if rng.rand() > 0.5:
                    momentum = np.cumsum(np.abs(rng.randn(20))) * 0.3
                    data[i, -20:, 0] += momentum
                    labels[i] = 1.0

        # Add noise
        data += rng.randn(*data.shape).astype(np.float32) * noise

        self.data   = torch.tensor(data)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────
#  2. METRICS
# ─────────────────────────────────────────────────────────

def binary_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Accuracy for binary classification (threshold at 0.5)."""
    predicted = (preds >= 0.5).float()
    return (predicted == labels).float().mean().item()


def compute_metrics(preds, labels):
    acc = binary_accuracy(preds, labels)
    # Avoid log(0)
    preds_clamp = preds.clamp(1e-7, 1 - 1e-7)
    loss = F.binary_cross_entropy(preds_clamp, labels).item()
    return {'accuracy': acc, 'loss': loss}


# ─────────────────────────────────────────────────────────
#  3. TRAINING LOOP
# ─────────────────────────────────────────────────────────

def train(
    model       : OvertureWithHead,
    domain      : str,
    train_loader: DataLoader,
    val_loader  : DataLoader,
    device      : torch.device,
    n_epochs    : int   = 30,
    lr          : float = 3e-4,
    patience    : int   = 8,
):
    """
    Full training loop with:
        - AdamW optimizer with weight decay
        - Cosine LR schedule
        - Early stopping
        - Per-epoch metrics
        - Loop usage tracking

    Args:
        model        : OvertureWithHead instance
        domain       : registered domain name
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        device       : cuda or cpu
        n_epochs     : maximum epochs
        lr           : initial learning rate
        patience     : early stopping patience
    """
    import torch.nn.functional as F

    optimizer = optim.AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = 1e-4,
        betas        = (0.9, 0.999),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = n_epochs,
        eta_min= lr * 0.01,
    )

    criterion = nn.BCELoss()

    best_val_acc  = 0.0
    best_epoch    = 0
    patience_count= 0
    history       = {
        'train_loss': [], 'train_acc': [],
        'val_loss'  : [], 'val_acc'  : [],
        'avg_loops' : [], 'lr'       : [],
    }

    print(f"\n{'═'*62}")
    print(f"  Training: {n_epochs} epochs max  |  LR: {lr}  |  Patience: {patience}")
    print(f"{'═'*62}")
    print(f"  {'Epoch':>5}  {'T-Loss':>8}  {'T-Acc':>7}  {'V-Loss':>8}  "
          f"{'V-Acc':>7}  {'Loops':>6}  {'LR':>8}")
    print(f"  {'─'*58}")

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()

        # ── Training phase ──
        model.train()
        train_losses, train_accs, loop_counts = [], [], []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            result = model(domain, x_batch)
            preds  = result['prediction']
            loss   = criterion(preds, y_batch)

            loss.backward()

            # Gradient clipping — important for complex networks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(binary_accuracy(preds.detach(), y_batch))
            loop_counts.append(result['loops_run'])

        # ── Validation phase ──
        model.eval()
        val_losses, val_accs = [], []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                result = model(domain, x_batch)
                preds  = result['prediction']
                loss   = criterion(preds, y_batch)

                val_losses.append(loss.item())
                val_accs.append(binary_accuracy(preds, y_batch))

        scheduler.step()

        # ── Compute epoch metrics ──
        t_loss = np.mean(train_losses)
        t_acc  = np.mean(train_accs)
        v_loss = np.mean(val_losses)
        v_acc  = np.mean(val_accs)
        avg_lp = np.mean(loop_counts)
        cur_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        history['avg_loops'].append(avg_lp)
        history['lr'].append(cur_lr)

        # ── Print epoch row ──
        improved = '★' if v_acc > best_val_acc else ' '
        print(f"  {epoch:>5}  {t_loss:>8.4f}  {t_acc:>6.1%}  {v_loss:>8.4f}  "
              f"{v_acc:>6.1%}  {avg_lp:>6.1f}  {cur_lr:>8.6f} {improved}")

        # ── Early stopping ──
        if v_acc > best_val_acc:
            best_val_acc   = v_acc
            best_epoch     = epoch
            patience_count = 0
            # Save best checkpoint
            torch.save({
                'epoch'     : epoch,
                'model_state': model.state_dict(),
                'val_acc'   : v_acc,
                'history'   : history,
            }, 'overture_best.pt')
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val acc: {best_val_acc:.1%} at epoch {best_epoch})")
                break

    print(f"  {'─'*58}")
    print(f"  Best validation accuracy: {best_val_acc:.1%} at epoch {best_epoch}")
    print(f"{'═'*62}\n")

    return history


# ─────────────────────────────────────────────────────────
#  4. EVALUATION — show predictions vs ground truth
# ─────────────────────────────────────────────────────────

def evaluate(model, domain, loader, device, n_show=16):
    """Show predictions vs ground truth for a batch."""
    model.eval()
    x_batch, y_batch = next(iter(loader))
    x_batch = x_batch.to(device)

    with torch.no_grad():
        result = model(domain, x_batch)

    preds  = result['prediction'].cpu()
    labels = y_batch.cpu()

    print(f"{'─'*50}")
    print(f"  Sample predictions (first {min(n_show, len(preds))}):")
    print(f"  {'Pred':>8}  {'Label':>6}  {'Correct':>8}  {'Confidence':>12}")
    print(f"  {'─'*46}")

    correct = 0
    for i in range(min(n_show, len(preds))):
        p   = preds[i].item()
        l   = labels[i].item()
        is_correct = (p >= 0.5) == (l >= 0.5)
        conf = max(p, 1-p)
        mark = '✓' if is_correct else '✗'
        if is_correct:
            correct += 1
        print(f"  {p:>8.4f}  {l:>6.1f}  {mark:>8}  {conf:>11.1%}")

    print(f"  {'─'*46}")
    print(f"  Accuracy on shown samples: {correct}/{min(n_show, len(preds))}")
    print(f"  Loops used: {result['loops_run']} / {model.backbone.core_loop.max_loops}")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Config ──
    SEQ_LEN     = 60
    FEATURE_DIM = 12
    BATCH_SIZE  = 32
    N_EPOCHS    = 40
    LR          = 3e-4
    PATTERN     = 'trend'    # try 'trend', 'spike', 'momentum'

    print(f"\nTask: binary classification — detect '{PATTERN}' pattern")
    print(f"Sequence: {SEQ_LEN} timesteps × {FEATURE_DIM} features")

    # ── Datasets ──
    train_dataset = SyntheticSequenceDataset(
        n_samples=2000, seq_len=SEQ_LEN, feature_dim=FEATURE_DIM,
        pattern=PATTERN, noise=0.5, seed=42
    )
    val_dataset = SyntheticSequenceDataset(
        n_samples=400, seq_len=SEQ_LEN, feature_dim=FEATURE_DIM,
        pattern=PATTERN, noise=0.5, seed=99
    )
    test_dataset = SyntheticSequenceDataset(
        n_samples=400, seq_len=SEQ_LEN, feature_dim=FEATURE_DIM,
        pattern=PATTERN, noise=0.5, seed=777
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)} samples  |  "
          f"Val: {len(val_dataset)}  |  Test: {len(test_dataset)}")

    # ── Model ──
    model = OvertureWithHead(
        d_model       = 64,
        k_sparse      = 16,
        n_heads       = 4,
        max_loops     = 8,
        ff_multiplier = 4,
        dropout       = 0.1,
        task          = 'binary',
        pool_strategy = 'attention',
        extract_mode  = 'learned',
    )
    model.register_domain('price', input_dim=FEATURE_DIM)
    model.to(device)

    params = model.count_parameters()
    print(f"\nModel parameters: {params['total']:,} total")

    # ── Train ──
    history = train(
        model        = model,
        domain       = 'price',
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
        n_epochs     = N_EPOCHS,
        lr           = LR,
        patience     = 10,
    )

    # ── Load best checkpoint ──
    checkpoint = torch.load('overture_best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} "
          f"(val acc: {checkpoint['val_acc']:.1%})\n")

    # ── Final evaluation on test set ──
    print("Final evaluation on held-out test set:")
    evaluate(model, 'price', test_loader, device, n_show=16)

    # ── Training summary ──
    print(f"Training summary:")
    print(f"  Peak train acc : {max(history['train_acc']):.1%}")
    print(f"  Peak val acc   : {max(history['val_acc']):.1%}")
    print(f"  Final avg loops: {history['avg_loops'][-1]:.1f} / 8")
    print(f"  Checkpoint saved: overture_best.pt")
    print(f"\nNext: wire up real price data from your CSV.")