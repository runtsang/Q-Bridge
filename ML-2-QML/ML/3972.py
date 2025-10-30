"""
Hybrid quantum–classical kernel and classifier factory.

The ML implementation uses PyTorch for a learnable RBF kernel and
TorchQuantum for a parameter‑free overlap kernel.  The two kernels are
combined into a single `QuantumKernelMethod` module that returns a weighted
sum of classical and quantum similarities.  A convenience `build_classifier_circuit`
function produces a fully‑connected network with metadata that mirrors the
quantum counterpart.
"""

from __future__ import annotations

from typing import Sequence, Iterable, Tuple, List
import numpy as np
import torch
from torch import nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

class ClassicalRBF(nn.Module):
    """Learnable radial‑basis‑function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * (diff * diff).sum(dim=-1, keepdim=True))

class QuantumOverlapKernel(tq.QuantumModule):
    """Quantum kernel that encodes data into Ry rotations and
    measures the overlap of the resulting states."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Simple encoding: Ry on each qubit with the feature value
        self.ansatz = [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(self.n_wires)
        ]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.q_device.reset_states(x.shape[0])

        # Encode x
        for info in self.ansatz:
            params = x[:, info["input_idx"]]
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

        # Encode y with negative parameters
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]]
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

        # Overlap measurement
        return torch.abs(self.q_device.states.view(-1)[0])

class QuantumKernelMethod(nn.Module):
    """Hybrid kernel combining classical RBF and quantum overlap."""
    def __init__(self, gamma: float = 1.0, n_wires: int = 4) -> None:
        super().__init__()
        self.rbf = ClassicalRBF(gamma)
        self.qkernel = QuantumOverlapKernel(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        rbf_val = self.rbf(x, y).squeeze()
        q_val = self.qkernel(x, y).squeeze()
        # Simple equal weighting; can be tuned
        return 0.5 * rbf_val + 0.5 * q_val

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor],
                      gamma: float = 1.0,
                      n_wires: int = 4) -> np.ndarray:
        model = QuantumKernelMethod(gamma, n_wires)
        return np.array([[model(x, y).item() for y in b] for x in a])

def build_classifier_circuit(num_features: int,
                             depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward network that mimics the structure of the quantum
    classifier: a stack of linear layers with ReLU activations followed by a
    2‑class head.  Returns the network, indices used for data encoding,
    per‑layer weight counts, and observables (here simply the class indices).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

__all__ = ["ClassicalRBF",
           "QuantumOverlapKernel",
           "QuantumKernelMethod",
           "build_classifier_circuit"]
