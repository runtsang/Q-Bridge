import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuantumNAT(nn.Module):
    """
    Classical hybrid model that merges a standard CNN backbone with a 2×2 patch
    convolution inspired by the quanvolution filter. The two representations
    are concatenated and fed into a fully‑connected classifier.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 2×2 patch convolution (quanvolution style)
        self.patch_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Fully connected head
        # After cnn: 16*7*7 = 784, after patch_conv: 4*14*14 = 784
        self.fc = nn.Sequential(
            nn.Linear(784 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        cnn_feat = self.cnn(x).view(x.size(0), -1)
        patch_feat = self.patch_conv(x).view(x.size(0), -1)
        # Concatenate
        combined = torch.cat([cnn_feat, patch_feat], dim=1)
        logits = self.fc(combined)
        return self.norm(logits)

__all__ = ["HybridQuantumNAT"]
