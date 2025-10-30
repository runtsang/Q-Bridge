import torch
from torch import nn
import torch.nn.functional as F

class QCNNModelEnhanced(nn.Module):
    """Fully connected network mirroring an adaptive QCNN architecture.

    This model accepts an 8‑dimensional input and expands the feature map
    through an optional depth parameter. Every convolution‑like block
    doubles the hidden size, while each pooling‑like block halves it.
    The architecture can be tuned for different input regimes.
    """

    def __init__(self, depth: int = 2, hidden_size: int = 16) -> None:
        super().__init__()
        self.depth = depth
        self.hidden_size = hidden_size

        # Feature‑map block
        layers = [nn.Linear(8, hidden_size), nn.Tanh(), nn.BatchNorm1d(hidden_size)]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.BatchNorm1d(hidden_size)])
        self.feature_map = nn.Sequential(*layers)

        # Convolution‑pooling blocks
        conv_blocks = []
        current_size = hidden_size
        for _ in range(depth):
            conv_blocks.append(nn.Sequential(
                nn.Linear(current_size, current_size),
                nn.Tanh(),
                nn.BatchNorm1d(current_size),
            ))
            current_size = max(1, current_size // 2)  # pooling reduces dimension
        self.conv_blocks = nn.ModuleList(conv_blocks)

        # Final classifier
        self.head = nn.Linear(current_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for conv in self.conv_blocks:
            x = conv(x)
        out = self.head(x)
        return F.log_softmax(out, dim=1)

def QCNNModelEnhancedFactory(depth: int = 2, hidden_size: int = 16) -> QCNNModelEnhanced:
    """Return a new instance of QCNNModelEnhanced."""
    return QCNNModelEnhanced(depth=depth, hidden_size=hidden_size)

__all__ = ["QCNNModelEnhanced", "QCNNModelEnhancedFactory"]
