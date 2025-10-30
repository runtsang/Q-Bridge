"""Hybrid sampler and estimator neural network combining SamplerQNN and EstimatorQNN architectures."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerEstimatorQNN(nn.Module):
    """
    A PyTorch module that shares a backbone network and provides two output heads:
    * sampler_head: softmax probabilities over 2 classes.
    * estimator_head: scalar regression output.
    The backbone mimics the combined depth of SamplerQNN and EstimatorQNN.
    """

    def __init__(self) -> None:
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
        )
        # Sampler head: output 2 probabilities
        self.sampler_head = nn.Linear(4, 2)
        # Estimator head: output single regression value
        self.estimator_head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning a tuple (probabilities, regression).
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (..., 2).
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            probabilities : shape (..., 2)
            regression : shape (..., 1)
        """
        x = self.backbone(inputs)
        probs = F.softmax(self.sampler_head(x), dim=-1)
        reg = self.estimator_head(x)
        return probs, reg

def SamplerQNN() -> HybridSamplerEstimatorQNN:
    """
    Compatibility wrapper that returns an instance of HybridSamplerEstimatorQNN.
    """
    return HybridSamplerEstimatorQNN()

__all__ = ["HybridSamplerEstimatorQNN", "SamplerQNN"]
