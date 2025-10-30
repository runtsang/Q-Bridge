from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Tuple

class HybridMLClassifier(nn.Module):
    """
    Purely classical feed‑forward network that mimics the structure of
    the original `build_classifier_circuit`.  It consists of `depth`
    hidden layers, each of size `num_features`, followed by a linear
    head producing `n_qubits` outputs.  The final activation is
    `tanh`, matching the behaviour of the fully‑connected layer
    example.
    """
    def __init__(self, num_features: int, depth: int, n_qubits: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, n_qubits))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def run(self, x: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper that accepts a NumPy array, converts it to a
        torch tensor, performs a forward pass, and returns a NumPy array.
        """
        with torch.no_grad():
            out = self.forward(torch.as_tensor(x, dtype=torch.float32))
        return out.detach().numpy()

def get_ml_metadata(num_features: int, depth: int, n_qubits: int) -> Tuple[list[int], list[int]]:
    """
    Return two lists: the sizes of each weight matrix + bias and the
    indices of the output qubits that the network will drive.
    """
    weight_sizes = []
    in_dim = num_features
    for _ in range(depth):
        weight_sizes.append(num_features * in_dim + num_features)
        in_dim = num_features
    weight_sizes.append(n_qubits * in_dim + n_qubits)
    return weight_sizes, list(range(n_qubits))
