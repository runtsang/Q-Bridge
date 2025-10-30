import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """
    Extended classical CNN‑fully connected model with residual connections,
    deeper feature extraction and channel‑wise attention.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 4, hidden_dim: int = 64):
        super().__init__()
        # Feature extractor with 3 conv blocks + residual
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        # Residual skip: 1x1 conv to match channels
        self.residual = nn.Conv2d(in_channels, 32, kernel_size=1, bias=False)
        # Attention module: simple channel‑wise attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 32 // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32 // 8, 32, kernel_size=1),
            nn.Sigmoid()
        )
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # Residual addition
        out = out + self.residual(x)
        # Attention weighting
        out = out * self.attention(out)
        bsz = out.shape[0]
        out = out.view(bsz, -1)
        out = self.fc(out)
        return self.norm(out)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
        return logits

    def score(self, x: torch.Tensor, y: torch.Tensor) -> float:
        logits = self.predict(x)
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item()
