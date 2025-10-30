from __future__ import annotations

import torch
from torch import nn
from Conv import Conv
from EstimatorQNN import EstimatorQNN

class HybridEstimatorQNN(nn.Module):
    """
    Hybrid classical estimator that combines a convolutional filter with a
    feed‑forward regressor.

    The class exposes the same EstimatorQNN API but adds a pretrained Conv
    filter as a feature extractor.  The filter output and a trainable weight
    are concatenated and fed into the EstimatorQNN network.  The architecture
    is fully differentiable and can be trained end‑to‑end with standard
    PyTorch optimisers.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        # Feature extractor
        self.conv = Conv(kernel_size=kernel_size, threshold=threshold)
        # Estimator network
        self.estimator = EstimatorQNN()
        # Trainable weight that can be tuned during training
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Input tensor with shape (..., kernel_size, kernel_size) that will
            be flattened and passed to the Conv filter.

        Returns
        -------
        torch.Tensor
            Regression output from the EstimatorQNN network.
        """
        # Run convolutional filter – returns a scalar
        conv_out = self.conv.run(data.detach().cpu().numpy())
        # Build input vector [conv, weight]
        inp = torch.tensor([conv_out, self.weight.item()], device=data.device, dtype=data.dtype).unsqueeze(0)
        # Pass through estimator network
        return self.estimator(inp)

__all__ = ["HybridEstimatorQNN"]
