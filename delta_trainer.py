import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SmallCNN
from data import get_task_data


class DeltaTrainer:
    def __init__(self, ewc_lambda=5000):
        self.model = SmallCNN()
        self.ewc_lambda = ewc_lambda

        # These get filled after each task
        self.fisher = {}    # importance of each parameter
        self.theta_ref = {} # snapshot of params after last task

    def compute_fisher(self, task_id):
        """
        Compute diagonal Fisher Information Matrix.
        This tells us WHICH parameters matter for old tasks.
        High Fisher = important = don't move this parameter much.
        """
        print("  Computing Fisher Information...")
        loader = DataLoader(
            get_task_data(task_id, train=True),
            batch_size=64, shuffle=True
        )

        # Reset Fisher accumulator
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        count = 0

        for images, labels in loader:
            if count >= 400:  # 400 samples is enough
                break

            self.model.zero_grad()
            outputs = self.model(images)

            # Use log probability of predicted class
            log_probs = nn.functional.log_softmax(outputs, dim=1)
            predicted = outputs.argmax(dim=1)
            log_probs[range(len(predicted)), predicted].mean().backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

            count += len(images)

        # Normalize and store
        self.fisher = {name: f / count for name, f in fisher.items()}

        # Save a snapshot of current parameters
        self.theta_ref = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }

        print(f"  Fisher computed over {count} samples.")

    def ewc_penalty(self):
        """
        Penalize moving away from theta_ref,
        weighted by how important each parameter is (Fisher).
        """
        if not self.fisher:
            return 0  # no penalty on very first task

        penalty = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                penalty += (
                    self.fisher[name] * (param - self.theta_ref[name]) ** 2
                ).sum()

        return (self.ewc_lambda / 2) * penalty

    def delta_update(self, task_id, epochs=3):
        """
        Train ONLY on new task data.
        EWC penalty prevents forgetting old tasks.
        """
        loader = DataLoader(
            get_task_data(task_id, train=True),
            batch_size=64, shuffle=True
        )

        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in loader:
                optimizer.zero_grad()

                new_loss = criterion(self.model(images), labels)
                ewc_loss = self.ewc_penalty()
                total_loss_val = new_loss + ewc_loss

                total_loss_val.backward()
                optimizer.step()
                total_loss += new_loss.item()

            avg_loss = total_loss / len(loader)
            print(f"  Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")


# Sanity check
if __name__ == "__main__":
    from trainer import evaluate

    trainer = DeltaTrainer(ewc_lambda=5000)

    print("=== Task 0 ===")
    trainer.delta_update(task_id=0, epochs=2)
    trainer.compute_fisher(task_id=0)
    acc0 = evaluate(trainer.model, task_id=0)
    print(f"Accuracy on Task 0: {acc0:.2%}")

    print("\n=== Task 1 (delta only, no retraining) ===")
    trainer.delta_update(task_id=1, epochs=2)
    trainer.compute_fisher(task_id=1)
    acc0_after = evaluate(trainer.model, task_id=0)
    acc1 = evaluate(trainer.model, task_id=1)
    print(f"Accuracy on Task 0 after learning Task 1: {acc0_after:.2%}")
    print(f"Accuracy on Task 1: {acc1:.2%}")
    print("\nDelta trainer works!")