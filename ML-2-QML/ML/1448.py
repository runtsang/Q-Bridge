import torch
import torch.nn as nn
import torch.nn.functional as F

class Hybrid(nn.Module):
    """
    Classical hybrid layer that replaces a quantum expectation head.

    The layer applies a learnable linear transform followed by a
    parameterised sigmoid activation.  The shift and scale are
    learnable parameters, allowing the network to adapt the decision
    boundary during training while keeping the interface identical
    to the original quantum head.
    """

    def __init__(self, in_features: int, shift: float = 0.0, scale: float = 1.0, learnable: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = nn.Parameter(torch.tensor(shift)) if learnable else shift
        self.scale = nn.Parameter(torch.tensor(scale)) if learnable else scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear transform
        logits = self.linear(x)
        # Parameterised sigmoid
        logits = logits * self.scale + self.shift
        probs = torch.sigmoid(logits)
        # Return 2‑class probability distribution
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["Hybrid"]
