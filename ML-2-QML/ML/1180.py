import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence

class FCL(nn.Module):
    """
    Multi‑layer fully‑connected neural network that emulates a quantum
    fully‑connected layer but with classical depth, batch‑normalisation
    and dropout.  The ``run`` method reshapes a flat list of
    parameters into the network weights and biases, performs a forward
    pass on a dummy input and returns the mean tanh activation as an
    expectation value.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_dims: Sequence[int] = (32, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Accept a flat list of parameters, inject them into the network,
        run a forward pass on a dummy input and return the mean tanh
        activation as a 1‑element numpy array.
        """
        # Flatten and validate
        params = torch.tensor(list(thetas), dtype=torch.float32)
        expected = sum(p.numel() for p in self.parameters())
        if params.numel()!= expected:
            raise ValueError(f"Expected {expected} parameters, got {params.numel()}")
        # Assign parameters
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data = params[idx : idx + n].view_as(p).data.clone()
            idx += n
        # Dummy input
        x = torch.randn(1, self.net[0].in_features)
        out = self.net(x)
        expectation = out.mean().item()
        return np.array([expectation])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run the network on a real input vector ``x`` and return the
        scalar output.  ``x`` should be a 1‑D array of length
        ``n_features``.
        """
        self.eval()
        with torch.no_grad():
            inp = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            out = self.net(inp)
            return out.squeeze().numpy()
