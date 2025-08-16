# src/kws/models/cnn.py
import torch
import torch.nn as nn

class TinyCNN(nn.Module):
    """
    Input:  X [B, 40, T]  (MFCC features)
    Output: logits [B, n_classes]
    """
    def __init__(self, n_mfcc: int = 40, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 40, T] -> [B, 1, 40, T]
        x = x.unsqueeze(1)
        z = self.features(x).flatten(1)   # [B, 64]
        return self.classifier(z)
