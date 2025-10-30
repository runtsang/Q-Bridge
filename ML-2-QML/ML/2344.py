import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuanvolutionClassifier(nn.Module):
    """
    Classical hybrid model that mimics the structure of a quanvolution filter
    followed by a QCNN-inspired fully‑connected head.
    """
    def __init__(self) -> None:
        super().__init__()
        # Patch‑level convolution (2x2 patches, stride 2)
        self.patch_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Additional feature extraction layers
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        # Fully‑connected head (QCNN‑style)
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Patch extraction
        x = F.relu(self.patch_conv(x))
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # Flatten
        x = x.view(x.size(0), -1)
        # QCNN‑style head
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionClassifier"]
