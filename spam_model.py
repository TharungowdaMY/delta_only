import torch
import torch.nn as nn


class SpamNet(nn.Module):
    """
    Simple neural network for spam detection.
    Input:  20 email features
    Output: 2 classes (spam or legitimate)
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


# Sanity check
if __name__ == "__main__":
    model  = SpamNet()
    dummy  = torch.randn(8, 20)
    output = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print("SpamNet works!")