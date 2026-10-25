import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from model import SmallCNN
from data import get_task_data
from replay_buffer import ReplayBuffer


class DeltaTrainer:
    def __init__(self, ewc_lambda=1000000,
                 buffer_size=500,
                 replay_ratio=0.2):
        self.model        = SmallCNN()
        self.ewc_lambda   = ewc_lambda
        self.fisher       = {}
        self.theta_ref    = {}
        self.buffer       = ReplayBuffer(max_size=buffer_size)
        self.replay_ratio = replay_ratio

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
            outputs   = self.model(images)
            log_probs = nn.functional.log_softmax(
                outputs, dim=1
            )
            predicted = outputs.argmax(dim=1)
            log_probs[range(len(predicted)),
                      predicted].mean().backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2
            count += len(images)

        self.fisher = {
            n: f / count for n, f in fisher.items()
        }
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
        new_data = get_task_data(task_id, train=True)

        # Add to replay buffer
        self.buffer.add_task_data(new_data)

        # Mix new + replay
        if not self.buffer.is_empty() and self.replay_ratio > 0:
            combined = ConcatDataset([new_data, self.buffer])
            loader   = DataLoader(
                combined, batch_size=64, shuffle=True
            )
            print(f"  Training on {len(new_data)} new + "
                  f"{len(self.buffer)} replay samples")
        else:
            loader = DataLoader(
                new_data, batch_size=64, shuffle=True
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
                new_loss = criterion(
                    self.model(images), labels
                )
                total_loss_val = new_loss + self.ewc_penalty()
                total_loss_val.backward()
                optimizer.step()
                total_loss += new_loss.item()

            avg_loss = total_loss / len(loader)
            print(f"  Epoch {epoch+1}/{epochs} "
                  f"- loss: {avg_loss:.4f}")