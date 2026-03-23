import torch
import numpy as np
from torch.utils.data import DataLoader
from data import get_task_data


def compute_drift_score(old_task_id, new_task_id,
                        n_samples=200):
    """
    Computes how different new task data is
    from old task data.
    Returns a score between 0 and 1.
    0 = identical distributions
    1 = completely different
    """
    # Load samples from both tasks
    old_loader = DataLoader(
        get_task_data(old_task_id, train=True),
        batch_size=n_samples, shuffle=True
    )
    new_loader = DataLoader(
        get_task_data(new_task_id, train=True),
        batch_size=n_samples, shuffle=True
    )

    # Get one batch from each
    old_images, _ = next(iter(old_loader))
    new_images, _ = next(iter(new_loader))

    # Flatten images to 1D vectors
    old_flat = old_images.view(
        old_images.size(0), -1
    ).numpy()
    new_flat = new_images.view(
        new_images.size(0), -1
    ).numpy()

    # Compute mean difference per pixel
    old_mean = old_flat.mean(axis=0)
    new_mean = new_flat.mean(axis=0)

    # Normalized drift score
    diff      = np.abs(old_mean - new_mean)
    max_range = 2.0  # pixel values range from -1 to 1
    drift     = float(diff.mean() / max_range)

    # Clip to [0, 1]
    return min(max(drift, 0.0), 1.0)


def compute_auto_lambda(drift_score,
                        lambda_min=100000,
                        lambda_max=5000000):
    """
    Maps drift score to lambda value.

    Low drift  → lambda close to lambda_min
    High drift → lambda close to lambda_max

    Uses exponential scaling so small drifts
    get small lambdas and large drifts get
    very large lambdas.
    """
    # Exponential interpolation
    log_min = np.log10(lambda_min)
    log_max = np.log10(lambda_max)
    log_lam = log_min + drift_score * (log_max - log_min)
    lam     = 10 ** log_lam

    return int(lam)


def get_adaptive_lambda(old_task_id, new_task_id,
                        verbose=True):
    """
    Full pipeline: compute drift → compute lambda.
    Returns the recommended lambda for this task transition.
    """
    if old_task_id is None:
        # First task — no old task to compare
        return 1000000

    drift = compute_drift_score(old_task_id, new_task_id)
    lam   = compute_auto_lambda(drift)

    if verbose:
        print(f"  Drift score:    {drift:.4f}")
        print(f"  Auto lambda:    {lam:,}")

        if drift < 0.1:
            print(f"  Assessment:     Low drift — "
                  f"learning freely")
        elif drift < 0.2:
            print(f"  Assessment:     Moderate drift — "
                  f"balanced protection")
        else:
            print(f"  Assessment:     High drift — "
                  f"strong protection")

    return lam


# Sanity check
if __name__ == "__main__":
    print("=== Auto Lambda Tuner Test ===\n")

    for new_task in range(1, 5):
        old_task = new_task - 1
        print(f"Task {old_task} → Task {new_task}:")
        lam = get_adaptive_lambda(old_task, new_task)
        print()

    print("Auto lambda tuner works!")