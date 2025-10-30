"""AdvancedSamplerQNN: a richer classical sampler network.

This class extends the original 2‑layer architecture with batch‑norm,
drop‑out and an extra hidden layer.  It also exposes a `sample`
method that draws from the categorical distribution defined by the
output probabilities, enabling direct use in generative pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedSamplerQNN(nn.Module):
    """A multi‑layer neural sampler with dropout and batch‑norm."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return class probabilities."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the network output.
        Returns a tensor of shape (num_samples, inputs.shape[0]) with integer labels.
        """
        probs = self.forward(inputs)
        samples = torch.multinomial(probs, num_samples, replacement=True)
        return samples.squeeze(-1)

__all__ = ["AdvancedSamplerQNN"]
