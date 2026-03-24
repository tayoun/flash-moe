#!/usr/bin/env python3
"""
train_predictor.py — Train and evaluate an expert routing predictor.

Reads binary routing data collected by `./infer --collect-routing FILE`,
trains a small MLP to predict which K experts will be activated from the
hidden state, and reports prediction accuracy per layer and overall.

Binary format per sample:
  int32  layer_idx
  int32  K
  float32[4096]  hidden_state (h_input: layer input before attention)
  int32[K]  expert_indices (actual routing result)

Usage:
    python train_predictor.py routing_data.bin [--epochs 20] [--hidden 256]
"""

import sys
import struct
import numpy as np
from pathlib import Path

HIDDEN_DIM = 4096
NUM_EXPERTS = 512
NUM_LAYERS = 60


def load_routing_data(path):
    """Load binary routing data into numpy arrays."""
    data = Path(path).read_bytes()
    offset = 0
    layers = []
    hiddens = []
    experts = []

    while offset < len(data) - 8:  # at least header
        layer_idx = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        K = struct.unpack_from('<i', data, offset)[0]
        offset += 4

        # hidden state: 4096 floats
        h = np.frombuffer(data, dtype=np.float32, count=HIDDEN_DIM, offset=offset).copy()
        offset += HIDDEN_DIM * 4

        # expert indices: K ints
        ei = np.frombuffer(data, dtype=np.int32, count=K, offset=offset).copy()
        offset += K * 4

        layers.append(layer_idx)
        hiddens.append(h)
        experts.append(ei)

    layers = np.array(layers, dtype=np.int32)
    hiddens = np.stack(hiddens)
    # Pad expert indices to max K
    max_K = max(len(e) for e in experts)
    experts_padded = np.zeros((len(experts), max_K), dtype=np.int32)
    for i, e in enumerate(experts):
        experts_padded[i, :len(e)] = e

    return layers, hiddens, experts_padded, max_K


def build_target_multilabel(expert_indices, num_experts=NUM_EXPERTS):
    """Convert expert indices to multi-label binary targets."""
    N = len(expert_indices)
    targets = np.zeros((N, num_experts), dtype=np.float32)
    for i in range(N):
        for j in range(expert_indices.shape[1]):
            targets[i, expert_indices[i, j]] = 1.0
    return targets


def train_and_evaluate(path, hidden_size=256, epochs=20, lr=1e-3, K_pred=4):
    """Train per-layer MLPs and evaluate prediction accuracy."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("ERROR: pip install torch")
        sys.exit(1)

    print(f"Loading routing data from {path}...")
    layers, hiddens, experts, K = load_routing_data(path)
    print(f"  {len(layers)} samples, K={K}, layers 0-{layers.max()}")
    print(f"  Hidden state shape: {hiddens.shape}")
    print(f"  Hidden RMS: {np.sqrt(np.mean(hiddens**2)):.4f}")

    # Analyze temporal locality baseline
    print("\n=== Temporal Locality Baseline ===")
    prev_experts = {}
    temporal_hits = 0
    temporal_total = 0
    for i in range(len(layers)):
        li = layers[i]
        ei = set(experts[i].tolist())
        if li in prev_experts:
            hits = len(ei & prev_experts[li])
            temporal_hits += hits
            temporal_total += K
        prev_experts[li] = ei
    if temporal_total > 0:
        print(f"  Temporal hit rate: {temporal_hits}/{temporal_total} = "
              f"{temporal_hits/temporal_total*100:.1f}%")

    # Split by layer
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nTraining on: {device}")

    # Train one shared model with layer embedding
    class ExpertPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_experts, num_layers):
            super().__init__()
            self.layer_emb = nn.Embedding(num_layers, 32)
            self.net = nn.Sequential(
                nn.Linear(input_dim + 32, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_experts),
            )

        def forward(self, x, layer_ids):
            le = self.layer_emb(layer_ids)
            combined = torch.cat([x, le], dim=-1)
            return self.net(combined)

    model = ExpertPredictor(HIDDEN_DIM, hidden_size, NUM_EXPERTS, NUM_LAYERS).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters ({param_count*4/1024/1024:.1f} MB)")

    # Prepare data
    targets = build_target_multilabel(experts, NUM_EXPERTS)

    # 80/20 split (by token, not by sample, to avoid leaking temporal info)
    n_tokens = len(layers) // NUM_LAYERS
    split = int(n_tokens * 0.8) * NUM_LAYERS
    train_idx = np.arange(split)
    test_idx = np.arange(split, len(layers))

    X_train = torch.tensor(hiddens[train_idx], dtype=torch.float32)
    L_train = torch.tensor(layers[train_idx], dtype=torch.long)
    Y_train = torch.tensor(targets[train_idx], dtype=torch.float32)

    X_test = torch.tensor(hiddens[test_idx], dtype=torch.float32)
    L_test = torch.tensor(layers[test_idx], dtype=torch.long)
    Y_test = torch.tensor(targets[test_idx], dtype=torch.float32)
    E_test = experts[test_idx]

    train_ds = TensorDataset(X_train, L_train, Y_train)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nTrain: {len(train_idx)} samples, Test: {len(test_idx)} samples")
    print(f"Training for {epochs} epochs...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, lb, yb in train_dl:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                test_logits = []
                for i in range(0, len(X_test), 512):
                    xb = X_test[i:i+512].to(device)
                    lb = L_test[i:i+512].to(device)
                    test_logits.append(model(xb, lb).cpu())
                test_logits = torch.cat(test_logits, dim=0).numpy()

            # Top-K prediction accuracy
            pred_indices = np.argsort(-test_logits, axis=1)[:, :K_pred]
            hits = 0
            total = 0
            for i in range(len(E_test)):
                actual = set(E_test[i].tolist())
                predicted = set(pred_indices[i].tolist())
                hits += len(actual & predicted)
                total += K
            acc = hits / total * 100 if total > 0 else 0

            avg_loss = total_loss / len(train_idx)
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}  "
                  f"top-{K_pred} hit rate={acc:.1f}% ({hits}/{total})")

    # Final detailed evaluation
    print("\n=== Final Evaluation ===")
    model.eval()
    with torch.no_grad():
        test_logits = []
        for i in range(0, len(X_test), 512):
            xb = X_test[i:i+512].to(device)
            lb = L_test[i:i+512].to(device)
            test_logits.append(model(xb, lb).cpu())
        test_logits = torch.cat(test_logits, dim=0).numpy()

    pred_indices = np.argsort(-test_logits, axis=1)[:, :K_pred]

    # Per-layer accuracy
    layer_hits = np.zeros(NUM_LAYERS)
    layer_total = np.zeros(NUM_LAYERS)
    for i in range(len(E_test)):
        li = layers[test_idx[i]]
        actual = set(E_test[i].tolist())
        predicted = set(pred_indices[i].tolist())
        layer_hits[li] += len(actual & predicted)
        layer_total[li] += K

    print(f"\nPer-layer top-{K_pred} hit rates:")
    for li in range(NUM_LAYERS):
        if layer_total[li] > 0:
            rate = layer_hits[li] / layer_total[li] * 100
            bar = '#' * int(rate / 2)
            print(f"  Layer {li:2d}: {rate:5.1f}% {bar}")

    overall_hits = int(layer_hits.sum())
    overall_total = int(layer_total.sum())
    overall_rate = overall_hits / overall_total * 100 if overall_total > 0 else 0
    print(f"\n  OVERALL: {overall_rate:.1f}% ({overall_hits}/{overall_total})")
    print(f"  Temporal baseline: {temporal_hits/temporal_total*100:.1f}%"
          if temporal_total > 0 else "  Temporal baseline: N/A")

    # Extended prediction: predict top-8 to cover more hits
    for k in [4, 6, 8, 12, 16]:
        pred_k = np.argsort(-test_logits, axis=1)[:, :k]
        hits = 0
        for i in range(len(E_test)):
            actual = set(E_test[i].tolist())
            predicted = set(pred_k[i].tolist())
            hits += len(actual & predicted)
        rate = hits / overall_total * 100
        print(f"  Top-{k:2d} predictions: {rate:.1f}% hit rate")

    # Compute potential speedup
    print("\n=== Speedup Estimate ===")
    baseline_io_ms = 2.4  # current expert_io per layer
    for hit_rate_pct in [overall_rate, 50, 60, 70, 80]:
        miss_rate = 1.0 - hit_rate_pct / 100.0
        # Hits are free (page cache warm), misses need sync pread
        # Miss pread time scales sub-linearly (parallel I/O)
        n_misses = miss_rate * K
        miss_io_ms = baseline_io_ms * (n_misses / K) ** 0.7  # sub-linear scaling
        pred_overhead_ms = 0.1  # ANE inference
        new_io_ms = pred_overhead_ms + miss_io_ms
        savings_ms = baseline_io_ms - new_io_ms
        new_total_layer = 4.0 - savings_ms  # ~4ms baseline total
        new_toks = 1000.0 / (new_total_layer * 60) if new_total_layer > 0 else 0
        label = "MEASURED" if hit_rate_pct == overall_rate else "target"
        print(f"  {hit_rate_pct:.0f}% hits ({label}): "
              f"expert_io {new_io_ms:.1f}ms → {new_toks:.1f} tok/s "
              f"(saves {savings_ms:.1f}ms/layer)")

    return model, overall_rate


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='Binary routing data file')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    train_and_evaluate(args.data_file, hidden_size=args.hidden,
                       epochs=args.epochs, lr=args.lr)
