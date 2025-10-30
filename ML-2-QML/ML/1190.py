"""QuantumHybridClassifier – classical surrogate for the hybrid quantum binary classifier.

The surrogate mirrors the quantum head of the original architecture but is
implemented entirely in PyTorch.  It exposes a learnable bias/shift
parameter, a small feed‑forward head, and an optional “quantum‑like”
activation that mimics the expectation value of a variational circuit.
This design allows quick ablation experiments without invoking a quantum
back‑end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumHybridSurrogate(nn.Module):
    """A lightweight classical head that emulates the quantum expectation layer.

    Parameters
    ----------
    in_features : int
        Size of the flattened feature vector from the CNN backbone.
    hidden_dim : int, default 32
        Width of the intermediate linear layer.
    shift : float, default 0.0
        Initial bias applied before the sigmoid.  Can be learned.
    """

    def __init__(self, in_features: int, hidden_dim: int = 32, shift: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.shift = nn.Parameter(torch.tensor(shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a two‑class probability distribution."""
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumHybridSurrogate"]
