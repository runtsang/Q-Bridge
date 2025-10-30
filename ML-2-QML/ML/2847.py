"""Hybrid fully connected and sampler module combining classical and quantum-inspired components."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class HybridFullyConnectedSampler(nn.Module):
    """
    A hybrid module that emulates a fully connected layer followed by a sampler network.
    The `run` method accepts a dictionary with keys:
        - 'fc_thetas': iterable of parameters for the linear layer.
        -'sampler_inputs': 2‑dim input for the sampler network.
        -'sampler_weights': 4‑dim weight vector for the sampler network.
    It returns a concatenated NumPy array of the linear expectation and the sampler
    softmax probabilities.
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        # Linear part
        self.linear = nn.Linear(n_features, 1)
        # Sampler part: simple linear mapping from concatenated inputs and weights
        self.sampler = nn.Sequential(
            nn.Linear(6, 2),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Linear expectation
        fc_thetas = params["fc_thetas"]
        linear_out = torch.tanh(self.linear(fc_thetas))
        # Sampler input
        sampler_inputs = params["sampler_inputs"]
        sampler_weights = params["sampler_weights"]
        # Concatenate inputs and weights
        concat = torch.cat([sampler_inputs, sampler_weights], dim=-1)
        sampler_out = self.sampler(concat)
        # Concatenate linear and sampler outputs
        return torch.cat([linear_out, sampler_out])

    def run(self, thetas: dict[str, Iterable[float]]) -> np.ndarray:
        # Convert to tensors
        fc_thetas = torch.as_tensor(thetas["fc_thetas"], dtype=torch.float32).view(-1, 1)
        sampler_inputs = torch.as_tensor(thetas["sampler_inputs"], dtype=torch.float32).view(-1, 2)
        sampler_weights = torch.as_tensor(thetas["sampler_weights"], dtype=torch.float32).view(-1, 4)
        output = self.forward(
            {
                "fc_thetas": fc_thetas,
                "sampler_inputs": sampler_inputs,
                "sampler_weights": sampler_weights,
            }
        )
        return output.detach().numpy()


__all__ = ["HybridFullyConnectedSampler"]
