import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from torch.utils.data import DataLoader, ConcatDataset
from spam_model import SpamNet
from spam_data import SpamDataset
from replay_buffer import ReplayBuffer


def evaluate_spam(model, wave_id):
    """Test accuracy on a spam wave."""
    model.eval()
    dataset = SpamDataset(wave_id, split='test')
    loader  = DataLoader(dataset, batch_size=256)

    correct = total = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            outputs   = model(X)
            predicted = outputs.argmax(dim=1)
            correct  += (predicted == y).sum().item()
            total    += len(y)
            all_preds.extend(predicted.tolist())
            all_labels.extend(y.tolist())

    acc = correct / total

    # Compute spam precision and recall
    tp = sum(1 for p, l in zip(all_preds, all_labels)
             if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels)
             if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels)
             if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall /
                (precision + recall)
                if (precision + recall) > 0 else 0)

    return acc, precision, recall, f1


def train_model(model, loader, epochs=5,
                ewc_fisher=None, ewc_theta=None,
                ewc_lambda=500000):
    """Train model with optional EWC penalty."""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)

            # Add EWC penalty if we have Fisher
            if ewc_fisher is not None:
                for name, param in model.named_parameters():
                    if name in ewc_fisher:
                        loss += (ewc_lambda / 2) * (
                            ewc_fisher[name] *
                            (param - ewc_theta[name]) ** 2
                        ).sum()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{epochs} "
              f"- loss: {avg:.4f}")


def compute_fisher_spam(model, loader, n_samples=400):
    """Compute Fisher Information for spam model."""
    fisher    = {n: torch.zeros_like(p)
                 for n, p in model.named_parameters()}
    theta_ref = {n: p.detach().clone()
                 for n, p in model.named_parameters()}

    model.eval()
    count = 0

    for X, y in loader:
        if count >= n_samples:
            break
        model.zero_grad()
        out       = model(X)
        log_probs = nn.functional.log_softmax(out, dim=1)
        predicted = out.argmax(dim=1)
        log_probs[range(len(predicted)),
                  predicted].mean().backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach() ** 2
        count += len(X)

    fisher = {n: f / count for n, f in fisher.items()}
    return fisher, theta_ref


# ── Main experiment ───────────────────────────────────
N_WAVES = 4
EPOCHS  = 5

print("=" * 55)
print("  SPAM DETECTION — DELTA TRAINING DEMO")
print("  Simulating evolving spam patterns over time")
print("=" * 55)

# Storage
retrain_results = []
delta_results   = []
naive_results   = []

retrain_time = 0
delta_time   = 0
naive_time   = 0

# Models
retrain_model = SpamNet()
delta_model   = SpamNet()
naive_model   = SpamNet()

# Delta components
ewc_fisher  = None
ewc_theta   = None
buffer      = ReplayBuffer(max_size=300)

all_wave_data = []  # for full retrain

for wave_id in range(N_WAVES):
    print(f"\n{'─'*55}")
    print(f" Wave {wave_id} — new spam pattern arriving")
    print(f"{'─'*55}")

    wave_data = SpamDataset(wave_id, split='train')
    all_wave_data.append(wave_data)
    wave_loader = DataLoader(
        wave_data, batch_size=64, shuffle=True
    )

    # ── 1. Full retrain ───────────────────────────────
    print("\n[Full retrain] retraining on all waves...")
    t0       = time.time()
    combined = ConcatDataset(all_wave_data)
    retrain_loader = DataLoader(
        combined, batch_size=64, shuffle=True
    )
    retrain_model = SpamNet()
    train_model(retrain_model, retrain_loader,
                epochs=EPOCHS)
    retrain_time += time.time() - t0

    # ── 2. Delta + EWC + Replay ───────────────────────
    print("\n[Delta + EWC + Replay] updating on new wave...")
    t0 = time.time()
    buffer.add_task_data(wave_data)

    if len(buffer) > 0:
        combined_delta = ConcatDataset(
            [wave_data, buffer]
        )
        delta_loader = DataLoader(
            combined_delta, batch_size=64, shuffle=True
        )
    else:
        delta_loader = wave_loader

    print(f"  Training on {len(wave_data)} new + "
          f"{len(buffer)} replay samples")

    train_model(delta_model, delta_loader,
                epochs=EPOCHS,
                ewc_fisher=ewc_fisher,
                ewc_theta=ewc_theta)

    # Update Fisher after training
    ewc_fisher, ewc_theta = compute_fisher_spam(
        delta_model, wave_loader
    )
    delta_time += time.time() - t0

    # ── 3. Naive delta ────────────────────────────────
    print("\n[Naive delta] updating on new wave only...")
    t0 = time.time()
    train_model(naive_model, wave_loader, epochs=EPOCHS)
    naive_time += time.time() - t0

    # ── Evaluate all waves seen so far ────────────────
    waves_seen = list(range(wave_id + 1))

    rt_accs = [evaluate_spam(retrain_model, w)
               for w in waves_seen]
    dt_accs = [evaluate_spam(delta_model,   w)
               for w in waves_seen]
    nv_accs = [evaluate_spam(naive_model,   w)
               for w in waves_seen]

    rt_acc  = sum(r[0] for r in rt_accs) / len(rt_accs)
    dt_acc  = sum(r[0] for r in dt_accs) / len(dt_accs)
    nv_acc  = sum(r[0] for r in nv_accs) / len(nv_accs)

    # F1 score on latest wave
    rt_f1 = rt_accs[-1][3]
    dt_f1 = dt_accs[-1][3]
    nv_f1 = nv_accs[-1][3]

    retrain_results.append({
        'acc': rt_acc, 'f1': rt_f1
    })
    delta_results.append({
        'acc': dt_acc, 'f1': dt_f1
    })
    naive_results.append({
        'acc': nv_acc, 'f1': nv_f1
    })

    print(f"\n  Average accuracy over "
          f"{len(waves_seen)} wave(s):")
    print(f"  Full retrain     : "
          f"{rt_acc:.2%} (F1={rt_f1:.3f})")
    print(f"  Delta+EWC+Replay : "
          f"{dt_acc:.2%} (F1={dt_f1:.3f}) "
          f"gap={abs(rt_acc-dt_acc):.2%}")
    print(f"  Naive delta      : "
          f"{nv_acc:.2%} (F1={nv_f1:.3f})")

# ── Final summary ─────────────────────────────────────
speedup = retrain_time / delta_time

print(f"\n{'='*55}")
print(" SPAM DETECTION FINAL RESULTS")
print(f"{'='*55}")
print(f"  Full retrain acc     : "
      f"{retrain_results[-1]['acc']:.2%}")
print(f"  Delta+EWC+Replay acc : "
      f"{delta_results[-1]['acc']:.2%}")
print(f"  Naive delta acc      : "
      f"{naive_results[-1]['acc']:.2%}")
print(f"\n  Full retrain F1      : "
      f"{retrain_results[-1]['f1']:.3f}")
print(f"  Delta+EWC+Replay F1  : "
      f"{delta_results[-1]['f1']:.3f}")
print(f"  Naive delta F1       : "
      f"{naive_results[-1]['f1']:.3f}")
print(f"\n  Training time:")
print(f"  Full retrain : {retrain_time:.1f}s")
print(f"  Delta+EWC    : {delta_time:.1f}s")
print(f"  Speedup      : {speedup:.1f}x faster")
print(f"{'='*55}")

# Save results
with open("spam_results.json", "w") as f:
    json.dump({
        "retrain_accs": [r['acc'] for r in retrain_results],
        "delta_accs":   [r['acc'] for r in delta_results],
        "naive_accs":   [r['acc'] for r in naive_results],
        "retrain_f1":   [r['f1'] for r in retrain_results],
        "delta_f1":     [r['f1'] for r in delta_results],
        "naive_f1":     [r['f1'] for r in naive_results],
        "retrain_time": retrain_time,
        "delta_time":   delta_time,
        "speedup":      speedup
    }, f)

print("\nResults saved to spam_results.json")