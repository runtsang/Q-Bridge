import torch
import torch.nn as nn
from torch.nn import functional as F

class QuantumClassifierModel(nn.Module):
    """
    A residual‑style neural network that mirrors the quantum interface.
    It accepts the same ``num_features`` and ``depth`` arguments as the original,
    but adds skip connections and a small bottleneck for richer feature
    extraction.  The output of this network is used as a classical
    embedding that can be fed into a quantum circuit or compared directly
    with the quantum expectation values during training.
    """

    def __init__(self, num_features: int, depth: int):
        super().__init__()
        self.num_features = num_features
        self.depth = depth

        # Build a residual block for each layer.
        self.blocks = nn.ModuleList()
        in_dim = num_features
        for _ in range(depth):
            out_dim = in_dim  # keep dimensionality the same for skip
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim),
                )
            )
            # skip connection after the linear → ReLU → linear
        self.head = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            residual = out
            out = block(out)
            out = out + residual  # add skip connection
            out = F.relu(out)
        # Final linear head for binary classification
        return self.head(out)
