import torch
import torch.nn as nn
import numpy as np
from typing import Iterable

class HybridFCL(nn.Module):
    """Classical hybrid fully‑connected layer that blends convolutional feature
    extraction, a deep fully‑connected stack, and a classification head.
    The architecture is a synthesis of the original FCL stub, the
    QFCModel, QCNNModel, and QuantumClassifierModel.

    Public API matches the original ``FCL`` factory:
        - ``run(thetas)`` accepts a 1‑D iterable of input features and
          returns the network output as a NumPy array.
    """

    def __init__(self, num_features: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        # Feature extractor – a lightweight 2‑D CNN (QFCModel style)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully‑connected stack (QCNNModel style)
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass for a batch of images or flattened vectors.

        Args:
            x: Tensor of shape (batch, 1, 28, 28) or (batch, 784).
                If a 1‑D tensor is provided, it is reshaped to a
                28×28 image.
        """
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        features = self.features(x)
        flattened = features.view(x.shape[0], -1)
        out = self.fc(flattened)
        return self.norm(out)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Compatibility wrapper: ``thetas`` is treated as a single example
        of input features.  The method returns the network output as a
        NumPy array.
        """
        inp = torch.tensor(list(thetas), dtype=torch.float32).view(1, 1, 28, 28)
        with torch.no_grad():
            out = self.forward(inp)
        return out.detach().cpu().numpy()

    def num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def weight_sizes(self) -> list[int]:
        """Return a list with the number of parameters per linear layer."""
        return [p.numel() for p in self.parameters() if p.requires_grad]

def FCL() -> HybridFCL:
    """Factory returning an instance configured like the original FCL."""
    return HybridFCL()

__all__ = ["HybridFCL", "FCL"]
