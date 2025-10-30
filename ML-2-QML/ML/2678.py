"""HybridSamplerQNN: Classical approximation of a quantum sampler with fully connected layer.

The class combines a simple neural sampler network with a linear fully connected layer,
mimicking the structure of a quantum sampler QNN and a fully connected quantum layer.
It can be used to benchmark the quantum implementation or as a baseline for
classical optimization.

Author: <Name>
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridSamplerQNN(nn.Module):
    """
    Classical neural network that emulates a quantum sampler followed by a fully
    connected layer. The network consists of:
    * sampler: a small feed‑forward network producing a probability distribution
      over two outputs.
    * fcl: a linear layer that maps the sampler probabilities to a scalar output,
      analogous to the expectation value of a quantum observable.
    """

    def __init__(self) -> None:
        super().__init__()
        # Sampler network: 2→4→2 (softmax)
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        # Fully connected layer: 2→1
        self.fcl = nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (..., 2) containing the two input features.

        Returns
        -------
        torch.Tensor
            Tensor of shape (..., 1) containing the scalar output.
        """
        probs = F.softmax(self.sampler(inputs), dim=-1)
        return self.fcl(probs)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Mimic the quantum ``run`` method: take a list of parameters and return the
        expectation value of the output layer.  Parameters are interpreted as
        weights for the fully connected layer.

        Parameters
        ----------
        thetas : np.ndarray
            1‑D array of length 1 (the weight for the linear layer).

        Returns
        -------
        np.ndarray
            Array containing the scalar expectation value.
        """
        if len(thetas)!= 1:
            raise ValueError("Expected a single weight for the fully connected layer.")
        # Create a dummy input of zeros
        dummy = torch.zeros(1, 2)
        probs = F.softmax(self.sampler(dummy), dim=-1)
        # manually apply the linear weight
        weight = torch.tensor(thetas[0], dtype=torch.float32).view(1, 1)
        bias = torch.zeros(1, 1)
        output = torch.matmul(probs, weight) + bias
        return output.detach().numpy()

__all__ = ["HybridSamplerQNN"]
