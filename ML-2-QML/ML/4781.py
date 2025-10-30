"""Hybrid classical self‑attention module that fuses a convolutional front‑end
with a multi‑head dot‑product attention mechanism.  The module is designed
to be drop‑in for the quantum SelfAttention class while providing richer
feature extraction and a batch‑friendly API."""
import numpy as np
import torch
from torch import nn

class HybridSelfAttention(nn.Module):
    """
    Classical self‑attention with a learnable 2×2 convolution filter.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the query/key/value vectors.
    kernel_size : int, default 2
        Size of the convolution kernel used as a pre‑processing step.
    threshold : float, default 0.0
        Threshold used in the sigmoid activation of the convolution output.
    """
    def __init__(self, embed_dim: int, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.threshold = threshold

    def forward(self, inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        """
        Compute a self‑attention weighted sum of the input features.

        Parameters
        ----------
        inputs : torch.Tensor
            Batch of feature maps of shape (batch, 1, H, W).
        rotation_params : np.ndarray
            Parameters used to generate the query vector.
        entangle_params : np.ndarray
            Parameters used to generate the key vector.

        Returns
        -------
        torch.Tensor
            Attention‑aggregated representations of shape (batch, embed_dim).
        """
        # Convolutional pre‑processing
        conv_out = self.conv(inputs)
        conv_out = torch.sigmoid(conv_out - self.threshold)
        conv_flat = conv_out.view(conv_out.size(0), -1)

        # Linear projections to form query/key/value
        q = torch.matmul(conv_flat, torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32))
        k = torch.matmul(conv_flat, torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32))
        v = conv_flat

        # Dot‑product attention
        scores = torch.softmax(q @ k.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

    def evaluate(self,
                 observables,
                 parameter_sets):
        """
        Helper that mimics the FastBaseEstimator evaluate interface.
        """
        from collections.abc import Iterable
        if not isinstance(observables, Iterable) or not observables:
            observables = [lambda out: out.mean(dim=-1)]

        results = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                rotation_params, entangle_params = params
                out = self.forward(torch.randn(1,1,2,2),  # dummy input
                                   np.array(rotation_params),
                                   np.array(entangle_params))
                row = []
                for obs in observables:
                    val = obs(out)
                    row.append(float(val.mean().cpu()))
                results.append(row)
        return results

__all__ = ["HybridSelfAttention"]
