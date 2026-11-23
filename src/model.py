import torch.nn as nn


class FruitCNN(nn.Module):
    """Own CNN: 3 conv blocks + 2 dense layers. Output = 1 logit (fresh=0, rotten=1)."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64x8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 1),  # raw logit; sigmoid is applied inside BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


if __name__ == "__main__":
    m = FruitCNN()
    total = 0
    for name, p in m.named_parameters():
        print(f"{name:25s} {tuple(p.shape)!s:18s} {p.numel()}")
        total += p.numel()
    print("Total trainable parameters:", total)
