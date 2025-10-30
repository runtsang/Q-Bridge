import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SamplerQNN(nn.Module):
    """
    Classical neural sampler that can optionally integrate quantum kernel features.
    The network maps a 2‑dimensional input to a probability distribution over 2 classes.
    An optional ``kernel_func`` can be supplied to augment the input with quantum‑derived
    similarity features before the final softmax.
    """
    def __init__(self, kernel_func: Optional[callable] = None):
        super().__init__()
        self.kernel_func = kernel_func
        # Core classifier
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param inputs: Tensor of shape (batch, 2).
        :return: Probabilities over 2 classes.
        """
        if self.kernel_func is not None:
            # Append kernel features along the feature dimension
            kernel_feats = self.kernel_func(inputs)
            if kernel_feats.ndim == 1:
                kernel_feats = kernel_feats.unsqueeze(-1)
            inputs = torch.cat([inputs, kernel_feats], dim=-1)
        return F.softmax(self.net(inputs), dim=-1)

    def set_kernel(self, kernel_func: callable):
        """Attach a quantum kernel function that accepts a tensor and returns similarity features."""
        self.kernel_func = kernel_func

__all__ = ["SamplerQNN"]
