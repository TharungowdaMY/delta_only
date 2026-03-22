from model import SmallCNN
from data import get_task_data
from trainer import evaluate
from delta_trainer import DeltaTrainer

for lam in [10000, 100000, 1000000]:
    trainer = DeltaTrainer(ewc_lambda=lam)

    trainer.delta_update(task_id=0, epochs=3)
    trainer.compute_fisher(task_id=0)

    acc0_before = evaluate(trainer.model, task_id=0)

    trainer.delta_update(task_id=1, epochs=3)

    acc0_after = evaluate(trainer.model, task_id=0)
    acc1 = evaluate(trainer.model, task_id=1)

    print(f"lambda={lam} | Task0 before: {acc0_before:.2%} | "
          f"Task0 after: {acc0_after:.2%} | Task1: {acc1:.2%}")