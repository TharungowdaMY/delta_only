import torch
import numpy as np
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    """
    Realistic spam dataset where:
    - Classes heavily overlap (hard to separate)
    - Each wave shifts the pattern significantly
    - Noise is high — mimics real email data
    """

    def __init__(self, wave_id, split='train',
                 n_samples=2000):
        np.random.seed(wave_id * 7)

        n_train = int(n_samples * 0.8)
        n_test  = n_samples - n_train
        n       = n_train if split == 'train' else n_test

        # Each wave rotates the decision boundary
        # making old patterns less useful
        angle = wave_id * (np.pi / 6)  # 30 degree rotation

        # Base features — heavily overlapping
        spam  = np.random.randn(n // 2, 20) * 1.5
        legit = np.random.randn(n // 2, 20) * 1.5

        # Small signal — easy to forget
        spam[:, 0]  += np.cos(angle) * 0.8
        spam[:, 1]  += np.sin(angle) * 0.8
        spam[:, 2]  -= np.cos(angle) * 0.6
        spam[:, 3]  += np.sin(angle) * 0.5
        spam[:, 4]  -= 0.4

        legit[:, 0] -= np.cos(angle) * 0.8
        legit[:, 1] -= np.sin(angle) * 0.8
        legit[:, 2] += np.cos(angle) * 0.6
        legit[:, 3] -= np.sin(angle) * 0.5
        legit[:, 4] += 0.4

        # Add heavy noise — mimics real world
        spam  += np.random.randn(n // 2, 20) * 1.2
        legit += np.random.randn(n // 2, 20) * 1.2

        # Combine
        X = np.vstack([spam, legit])
        y = np.array([1] * (n // 2) + [0] * (n // 2))

        # Shuffle
        idx  = np.random.permutation(len(y))
        X, y = X[idx], y[idx]

        # Normalize
        mean = X.mean(axis=0)
        std  = X.std(axis=0) + 1e-8
        X    = (X - mean) / std

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Sanity check
if __name__ == "__main__":
    print("=== Realistic Spam Dataset Test ===\n")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    for wave_id in range(4):
        train = SpamDataset(wave_id, split='train')
        test  = SpamDataset(wave_id, split='test')

        X_train = train.X.numpy()
        y_train = train.y.numpy()
        X_test  = test.X.numpy()
        y_test  = test.y.numpy()

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        acc = accuracy_score(
            y_test, clf.predict(X_test)
        )

        print(f"Wave {wave_id}: "
              f"train={len(train)} "
              f"test={len(test)} "
              f"baseline_acc={acc:.2%}")

    print("\nRealistic spam dataset works!")