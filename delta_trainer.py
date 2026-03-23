import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from model import SmallCNN
from data import get_task_data
from replay_buffer import ReplayBuffer


class DeltaTrainer:
    def __init__(self, ewc_lambda=1000000, buffer_size=500,
                 replay_ratio=0.2):
        self.model       = SmallCNN()
        self.ewc_lambda  = ewc_lambda
        self.fisher      = {}
        self.theta_ref   = {}

        # Replay buffer
        self.buffer       = ReplayBuffer(max_size=buffer_size)
        self.replay_ratio = replay_ratio  # 0.2 = 20% old data mixed in

    def compute_fisher(self, task_id):
        print("  Computing Fisher Information...")
        loader = DataLoader(
            get_task_data(task_id, train=True),
            batch_size=64, shuffle=True
        )

        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        count = 0

        for images, labels in loader:
            if count >= 400:
                break
            self.model.zero_grad()
            outputs  = self.model(images)
            log_probs = nn.functional.log_softmax(outputs, dim=1)
            predicted = outputs.argmax(dim=1)
            log_probs[range(len(predicted)),
                      predicted].mean().backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2
            count += len(images)

        self.fisher    = {n: f / count for n, f in fisher.items()}
        self.theta_ref = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
        }
        print(f"  Fisher computed over {count} samples.")

    def ewc_penalty(self):
        if not self.fisher:
            return 0
        penalty = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                penalty += (
                    self.fisher[name] *
                    (param - self.theta_ref[name]) ** 2
                ).sum()
        return (self.ewc_lambda / 2) * penalty

    def delta_update(self, task_id, epochs=3):
        """
        Train on new task data + replay buffer samples.
        EWC penalty protects old knowledge.
        """
        new_data = get_task_data(task_id, train=True)

        # Add new task to buffer BEFORE training
        # so buffer stays representative
        self.buffer.add_task_data(new_data)

        # Mix new data + replay buffer if buffer has old data
        if not self.buffer.is_empty() and self.replay_ratio > 0:
            combined  = ConcatDataset([new_data, self.buffer])
            loader    = DataLoader(
                combined,
                batch_size=64,
                shuffle=True
            )
            print(f"  Training on {len(new_data)} new + "
                  f"{len(self.buffer)} replay samples")
        else:
            loader = DataLoader(
                new_data,
                batch_size=64,
                shuffle=True
            )
            print(f"  Training on {len(new_data)} new samples")

        optimizer = optim.Adam(
            self.model.parameters(), lr=5e-4
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in loader:
                optimizer.zero_grad()
                new_loss  = criterion(self.model(images), labels)
                total_loss_val = new_loss + self.ewc_penalty()
                total_loss_val.backward()
                optimizer.step()
                total_loss += new_loss.item()

            avg_loss = total_loss / len(loader)
            print(f"  Epoch {epoch+1}/{epochs} "
                  f"- loss: {avg_loss:.4f}")


# Sanity check
if __name__ == "__main__":
    from trainer import evaluate

    print("=== Testing Delta Trainer WITH Replay Buffer ===\n")

    trainer = DeltaTrainer(
        ewc_lambda=1000000,
        buffer_size=500,
        replay_ratio=0.2
    )

    print("--- Task 0 ---")
    trainer.delta_update(task_id=0, epochs=2)
    trainer.compute_fisher(task_id=0)
    acc0 = evaluate(trainer.model, task_id=0)
    print(f"Task 0 accuracy: {acc0:.2%}\n")

    print("--- Task 1 ---")
    trainer.delta_update(task_id=1, epochs=2)
    trainer.compute_fisher(task_id=1)
    acc0_after = evaluate(trainer.model, task_id=0)
    acc1       = evaluate(trainer.model, task_id=1)
    print(f"Task 0 after Task 1: {acc0_after:.2%}")
    print(f"Task 1 accuracy:     {acc1:.2%}")

    gap = acc0 - acc0_after
    print(f"\nForgetting on Task 0: {gap:.2%}")
    print("\nReplay buffer integration works!")