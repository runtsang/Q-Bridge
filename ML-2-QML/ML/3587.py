import torch
import torch.nn as nn

class ConvFilter(nn.Module):
    """Convolutional filter emulating the quantum filter from Conv.py."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # Collapse spatial dimensions to produce a scalar per image
        return activations.mean(dim=[2, 3])

class BaseQFCModel(nn.Module):
    """Original QFCModel architecture from QuantumNAT.py."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

class QFCModel(BaseQFCModel):
    """Hybrid classical model that augments the base CNN with a lightweight
    convolutional filter.  The filter captures fine‑grained spatial patterns
    that are otherwise lost by pooling, and its scalar output is added to
    the final logits."""
    def __init__(self) -> None:
        super().__init__()
        self.conv_filter = ConvFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = super().forward(x)                 # shape (bsz, 4)
        filter_feat = self.conv_filter(x).unsqueeze(1)  # shape (bsz, 1)
        # Broadcast addition to match shape
        return base_out + filter_feat

__all__ = ["QFCModel"]
