import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNAT__gen004(nn.Module):
    """
    Extended classical model based on the original QFCModel.
    Adds a learnable pooling block (depthwise separable conv) and
    an optional classical post‑processing head which can be
    toggled via ``use_classical_head``.
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 pool_depthwise: bool = True,
                 use_classical_head: bool = True,
                 fc_hidden_dim: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.use_classical_head = use_classical_head

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Learnable pooling block
        if pool_depthwise:
            # depthwise separable conv for feature pooling
            self.pooling = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=1),
            )
        else:
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Flatten and fully connected head
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(16 * 1 * 1, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(fc_hidden_dim, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

        # Optional classical post‑processing head
        if self.use_classical_head:
            self.classical_post = nn.Sequential(
                nn.Linear(num_classes, num_classes),
                nn.ReLU(inplace=True),
                nn.Linear(num_classes, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.norm(x)
        if self.use_classical_head:
            x = self.classical_post(x)
        return x

__all__ = ["QuantumNAT__gen004"]
