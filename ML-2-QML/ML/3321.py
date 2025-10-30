import numpy as np
import torch
from torch import nn
from typing import Iterable

class HybridFullyConnectedLayer(nn.Module):
    """
    Classical implementation of a fully‑connected layer that mirrors
    the EstimatorQNN architecture but can be used as a drop‑in
    replacement for the original FCL.  The network consists of three
    linear blocks with tanh activations, matching the depth of the
    EstimatorQNN example, and exposes a `run` method that accepts a
    list of parameters and returns the mean activation – a convenient
    proxy for a quantum expectation value.
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Accepts a list of parameters, feeds them through the network
        and returns the mean activation as a numpy array.  This
        mirrors the behavior of the original FCL.run which returned a
        single expectation value.
        """
        values = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        output = self.net(values)
        expectation = torch.tanh(output).mean()
        return expectation.detach().numpy()

__all__ = ["HybridFullyConnectedLayer"]
