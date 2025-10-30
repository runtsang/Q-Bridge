from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalConvFilter(nn.Module):
    """Drop‑in classical replacement for a quanvolution filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W) or (1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2, 3))

class HybridHead(nn.Module):
    """Classical head that mimics the quantum expectation layer."""
    def __init__(self, in_features: int = 1, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x) + self.shift
        return torch.sigmoid(logits)

class HybridQCNet(nn.Module):
    """CNN based binary classifier that optionally uses a classical filter
    and a classical hybrid head."""
    def __init__(self, use_filter: bool = False, filter_kwargs: dict | None = None) -> None:
        super().__init__()
        self.use_filter = use_filter
        self.filter = ClassicalConvFilter(**(filter_kwargs or {})) if use_filter else None

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid_head = HybridHead(1, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_filter:
            # apply the classical filter to each channel independently
            filtered = []
            for c in range(x.shape[1]):
                channel = x[:, c:c+1, :, :]
                filtered.append(self.filter(channel))
            x = torch.stack(filtered, dim=1)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid_head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQCNet", "ClassicalConvFilter", "HybridHead"]
