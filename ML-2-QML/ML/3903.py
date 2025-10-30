"""Hybrid Sampler Quantum Neural Network – Classical implementation.

This class merges the two‑layer softmax sampler from SamplerQNN with the
expectation‑based fully‑connected layer from FCL.  It learns a mapping
from 2‑dimensional inputs to a probability distribution over two classes
while emulating the quantum expectation value with a classical approximation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridSamplerQNN(nn.Module):
    """
    Classical hybrid sampler.

    Architecture:
        1. Linear layer maps 2‑dim input to 4 weight parameters.
        2. Fully‑connected layer (tanh) processes these weights,
           producing a 2‑dim expectation vector.
        3. Softmax normalises the output to a probability distribution.
    """

    def __init__(self) -> None:
        super().__init__()
        # 2 → 4 mapping for quantum‑like weights
        self.weight_mapper = nn.Linear(2, 4)
        # 4 → 2 mapping that emulates the quantum expectation
        self.expectation_layer = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Tensor of shape (..., 2) – input features.

        Returns:
            Tensor of shape (..., 2) – softmax probabilities.
        """
        # Map to quantum‑like weights
        w = self.weight_mapper(inputs)
        # Classical approximation of the quantum expectation
        exp_val = torch.tanh(self.expectation_layer(w))
        # Softmax to obtain a probability distribution
        return F.softmax(exp_val, dim=-1)

    def sample(self, inputs: torch.Tensor, n_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the output distribution.

        Args:
            inputs: Tensor of shape (..., 2).
            n_samples: Number of samples per input.

        Returns:
            Numpy array of shape (n_samples, 2) with one‑hot encoded samples.
        """
        probs = self(inputs).detach().cpu().numpy()
        samples = np.random.choice(2, size=(n_samples, probs.shape[0]), p=probs.T)
        one_hot = np.eye(2)[samples]
        return one_hot
