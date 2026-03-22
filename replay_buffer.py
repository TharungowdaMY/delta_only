import random
import torch
from torch.utils.data import Dataset


class ReplayBuffer(Dataset):
    """
    Stores a fixed number of old samples.
    Uses reservoir sampling to stay representative
    of ALL tasks seen so far.
    """

    def __init__(self, max_size=500):
        self.max_size = max_size
        self.images = []
        self.labels = []
        self.total_seen = 0

    def add_task_data(self, dataset):
        """
        Add samples from a new task using
        reservoir sampling algorithm.
        """
        for img, label in dataset:
            self.total_seen += 1

            if len(self.images) < self.max_size:
                # Buffer not full yet — just add
                self.images.append(img)
                self.labels.append(label)
            else:
                # Buffer full — randomly replace
                # an existing sample
                replace_idx = random.randint(
                    0, self.total_seen - 1
                )
                if replace_idx < self.max_size:
                    self.images[replace_idx] = img
                    self.labels[replace_idx] = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def is_empty(self):
        return len(self.images) == 0


# Sanity check
if __name__ == "__main__":
    from data import get_task_data

    buffer = ReplayBuffer(max_size=500)

    # Add Task 0 data
    task0 = get_task_data(task_id=0, train=True)
    buffer.add_task_data(task0)

    print(f"After Task 0: buffer size = {len(buffer)}")
    print(f"Total seen = {buffer.total_seen}")

    # Add Task 1 data
    task1 = get_task_data(task_id=1, train=True)
    buffer.add_task_data(task1)

    print(f"After Task 1: buffer size = {len(buffer)}")
    print(f"Total seen = {buffer.total_seen}")

    # Check label distribution
    labels = [buffer.labels[i] for i in range(len(buffer))]
    zeros = labels.count(0)
    ones  = labels.count(1)
    print(f"Label distribution: 0={zeros}, 1={ones}")
    print("Replay buffer works!")