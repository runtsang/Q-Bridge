import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QuanvolutionHybrid(nn.Module):
    """Hybrid classical convolutional filter + classifier.

    This implementation deepens the original single‑layer filter with a small
    residual CNN and adds a multi‑task head that can predict both the digit
    class and a coarse rotation angle.  The class is deliberately lightweight
    so that it can be dropped into existing pipelines without changing the
    surrounding training logic.
    """
    def __init__(self, num_classes: int = 10, use_rotation_head: bool = False) -> None:
        super().__init__()
        self.use_rotation_head = use_rotation_head

        # 1 → 4 feature maps, 2×2 kernel, stride 2
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(4)

        # Residual block: 4 → 8 → 4
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4)

        # Flatten to 4 * 14 * 14 features
        self.flatten = nn.Flatten()

        # Digit classifier
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

        # Optional rotation classifier
        if use_rotation_head:
            self.rotation_head = nn.Linear(4 * 14 * 14, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        out = F.relu(self.bn1(self.conv1(x)))          # (B, 4, 14, 14)
        residual = out
        out = F.relu(self.bn2(self.conv2(out)))        # (B, 8, 14, 14)
        out = self.bn3(self.conv3(out))                # (B, 4, 14, 14)
        out = F.relu(out + residual)                   # Residual addition

        features = self.flatten(out)                   # (B, 4*14*14)
        logits = self.classifier(features)             # (B, num_classes)
        return F.log_softmax(logits, dim=-1)

    def predict_rotation(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_rotation_head:
            raise RuntimeError("Rotation head not enabled.")
        features = self.feature_extractor(x)
        logits = self.rotation_head(features)
        return F.log_softmax(logits, dim=-1)

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw feature vector before the classifier."""
        out = F.relu(self.bn1(self.conv1(x)))
        residual = out
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = F.relu(out + residual)
        return self.flatten(out)
