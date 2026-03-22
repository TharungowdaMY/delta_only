import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class TaskDataset(Dataset):
    def __init__(self, task_id, train=True):
        base = datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
        class_a = task_id * 2
        class_b = task_id * 2 + 1

        self.samples = []
        for img, label in base:
            if label == class_a:
                self.samples.append((img, 0))  # remap to 0
            elif label == class_b:
                self.samples.append((img, 1))  # remap to 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_task_data(task_id, train=True):
    return TaskDataset(task_id, train=train)


def get_task_loader(task_id, train=True, batch_size=64):
    return DataLoader(
        get_task_data(task_id, train=train),
        batch_size=batch_size,
        shuffle=train
    )


# Sanity check
if __name__ == "__main__":
    for task_id in range(5):
        loader = get_task_loader(task_id)
        images, labels = next(iter(loader))
        print(f"Task {task_id} | "
              f"labels: {labels.unique().tolist()} | "
              f"shape: {images.shape}")
    print("Data loading works!")