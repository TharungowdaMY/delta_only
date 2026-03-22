import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from model import SmallCNN
from data import get_task_data


def evaluate(model, task_id):
    """Test accuracy on a single task."""
    model.eval()
    dataset = get_task_data(task_id, train=False)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += len(labels)

    return correct / total


def full_retrain(all_task_ids, epochs=3):
    """Train a fresh model on ALL tasks seen so far."""
    model = SmallCNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Combine all task data
    combined = ConcatDataset([
        get_task_data(t, train=True) for t in all_task_ids
    ])
    loader = DataLoader(combined, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    return model


# Sanity check
if __name__ == "__main__":
    print("=== Testing full retraining on Task 0 only ===")
    model = full_retrain(all_task_ids=[0], epochs=2)

    acc = evaluate(model, task_id=0)
    print(f"\nAccuracy on Task 0: {acc:.2%}")
    print("Full retrainer works!")