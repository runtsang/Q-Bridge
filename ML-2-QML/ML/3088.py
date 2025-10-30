"""HybridNATModel: classical CNN with quantum‑inspired fully connected layer.

The model merges the convolutional feature extractor of the original
Quantum‑NAT with a fully‑connected layer that emulates a parameterised
quantum circuit (see FCL in reference 2).  The output is a 4‑dimensional
regression target.

This module is fully PyTorch‑compatible and can be dropped into any
training loop that expects an nn.Module.
"""

import torch
import torch.nn as nn
import numpy as np


# Classical fully‑connected layer mimicking a quantum circuit
class FCL:
    """Return an object with a ``run`` method mimicking the quantum example."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: list[float]) -> np.ndarray:
            values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    def __init__(self):
        self.layer = self.FullyConnectedLayer()

    def run(self, thetas: list[float]) -> np.ndarray:
        return self.layer.run(thetas)


class FCLWrapper(nn.Module):
    """Wrap the FCL class so it can be used as a PyTorch layer."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.fcl = FCL()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for sample in x:
            theta = sample.tolist()
            exp = self.fcl.run(theta)[0]
            outs.append(exp)
        return torch.tensor(outs, device=x.device, dtype=x.dtype).unsqueeze(1)


class HybridNATModel(nn.Module):
    """Classical CNN + quantum‑inspired fully‑connected layer."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fcl_wrapper = FCLWrapper(n_features=16 * 7 * 7)
        self.final_fc = nn.Linear(1, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        fcl_out = self.fcl_wrapper(flattened)  # shape (bsz,1)
        out = self.final_fc(fcl_out)          # shape (bsz,4)
        return self.norm(out)


__all__ = ["HybridNATModel"]
