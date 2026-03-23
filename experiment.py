import time
import json
from model import SmallCNN
from data import get_task_data
from trainer import evaluate, full_retrain
from delta_trainer import DeltaTrainer

N_TASKS = 5
EPOCHS  = 3

print("=" * 55)
print("   DELTA-ONLY TRAINING vs FULL RETRAINING")
print("   NOW WITH REPLAY BUFFER")
print("=" * 55)

retrain_accs = []
delta_accs   = []
naive_accs   = []

retrain_time = 0
delta_time   = 0
naive_time   = 0

# Delta + EWC + Replay buffer
delta_trainer = DeltaTrainer(
    ewc_lambda=1000000,
    buffer_size=500,
    replay_ratio=0.2
)

# Naive — no EWC no replay
naive_trainer = DeltaTrainer(
    ewc_lambda=0,
    buffer_size=0,
    replay_ratio=0.0
)

for task_id in range(N_TASKS):
    print(f"\n{'─'*55}")
    print(f" Task {task_id}  "
          f"(classes {task_id*2} & {task_id*2+1})")
    print(f"{'─'*55}")

    # 1. Full retrain
    print("[Full retrain] training on all tasks so far...")
    t0 = time.time()
    from torch.utils.data import ConcatDataset
    retrain_model = full_retrain(
        all_task_ids=list(range(task_id + 1)),
        epochs=EPOCHS
    )
    retrain_time += time.time() - t0

    # 2. Delta + EWC + Replay
    print("[Delta + EWC + Replay] updating on new task...")
    t0 = time.time()
    delta_trainer.delta_update(task_id=task_id, epochs=EPOCHS)
    delta_trainer.compute_fisher(task_id=task_id)
    delta_time += time.time() - t0

    # 3. Naive
    print("[Naive delta] updating on new task...")
    t0 = time.time()
    naive_trainer.delta_update(task_id=task_id, epochs=EPOCHS)
    naive_time += time.time() - t0

    # Evaluate all tasks seen so far
    tasks_seen = list(range(task_id + 1))

    rt_acc = sum(
        evaluate(retrain_model, t) for t in tasks_seen
    ) / len(tasks_seen)

    ew_acc = sum(
        evaluate(delta_trainer.model, t) for t in tasks_seen
    ) / len(tasks_seen)

    nv_acc = sum(
        evaluate(naive_trainer.model, t) for t in tasks_seen
    ) / len(tasks_seen)

    retrain_accs.append(rt_acc)
    delta_accs.append(ew_acc)
    naive_accs.append(nv_acc)

    print(f"\n  Average accuracy over {len(tasks_seen)} task(s):")
    print(f"  Full retrain     : {rt_acc:.2%}")
    print(f"  Delta+EWC+Replay : {ew_acc:.2%}  "
          f"(gap: {abs(rt_acc-ew_acc):.2%})")
    print(f"  Naive delta      : {nv_acc:.2%}")

# Final summary
print(f"\n{'='*55}")
print(" FINAL RESULTS")
print(f"{'='*55}")
print(f"  Full retrain final acc     : {retrain_accs[-1]:.2%}")
print(f"  Delta+EWC+Replay final acc : {delta_accs[-1]:.2%}")
print(f"  Naive delta final acc      : {naive_accs[-1]:.2%}")
print(f"\n  Gap vs full retrain: "
      f"{abs(retrain_accs[-1]-delta_accs[-1]):.2%}")
print(f"\n  Total training time:")
print(f"  Full retrain : {retrain_time:.1f}s")
print(f"  Delta+EWC    : {delta_time:.1f}s")
print(f"  Speedup      : {retrain_time/delta_time:.1f}x faster")
print(f"{'='*55}")

# Save results
with open("results.json", "w") as f:
    json.dump({
        "retrain_accs": retrain_accs,
        "delta_accs":   delta_accs,
        "naive_accs":   naive_accs,
        "retrain_time": retrain_time,
        "delta_time":   delta_time,
        "naive_time":   naive_time
    }, f)

print("\nResults saved to results.json")
print("Run: streamlit run app.py to see updated demo!")