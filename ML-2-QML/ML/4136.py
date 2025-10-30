import torch
from torch import nn
import numpy as np

from.FastBaseEstimator import FastEstimator
from.SelfAttention import SelfAttention

class ConvHybrid(nn.Module):
    """Hybrid classical convolution + self‑attention block with fast evaluation."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, embed_dim: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.attention = SelfAttention(embed_dim=embed_dim)
        # FastEstimator will wrap this model for batched inference
        self.estimator = FastEstimator(self)

    def forward(self, data: np.ndarray) -> float:
        """
        Run a single data point through convolution and attention.
        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).
        Returns
        -------
        float
            Scalar output after attention aggregation.
        """
        # Convolution
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits = self.conv(tensor)
        conv_out = torch.sigmoid(logits - self.threshold).squeeze()

        # Flatten and feed into self‑attention
        flat = conv_out.view(-1).numpy()
        # Dummy rotation/entangle params for demonstration
        rot = np.random.randn(self.attention.embed_dim, self.attention.embed_dim)
        ent = np.random.randn(self.attention.embed_dim, self.attention.embed_dim)
        att_out = self.attention.run(rot, ent, flat)
        return att_out.mean()

    def evaluate(self,
                 observables,
                 parameter_sets,
                 shots: int | None = None,
                 seed: int | None = None):
        """
        Delegate evaluation to FastEstimator, adding optional shot noise.
        """
        return self.estimator.evaluate(observables, parameter_sets,
                                       shots=shots, seed=seed)

def Conv() -> ConvHybrid:
    """Convenience factory matching the original Conv.py interface."""
    return ConvHybrid()

__all__ = ["ConvHybrid", "Conv"]
