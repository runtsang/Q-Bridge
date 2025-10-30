"""Hybrid quantum‑classical classifier combining a quantum kernel and a classical read‑out.

This module implements a two‑stage pipeline:
1.  A TorchQuantum ansatz evaluates a quantum kernel between the input
    and a fixed set of support vectors, producing a feature vector.
2.  A simple feed‑forward network maps the kernel feature vector to
    class logits.

The design mirrors the classical counterpart but replaces the RBF kernel
with a parameter‑free quantum kernel based on a variational ansatz.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Iterable, Tuple

# ---------- Quantum kernel ----------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encodes two inputs into a quantum device and computes their overlap."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode first argument
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Un‑encode second argument with negative parameters
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Fixed TorchQuantum ansatz used as a quantum kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return the absolute value of the first element of the state vector.
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ---------- Classical classifier ---------------------------------------------
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a simple feed‑forward network that consumes the quantum kernel."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# ---------- Hybrid model ------------------------------------------------------
class SharedClassName(nn.Module):
    """Hybrid quantum‑classical classifier using a quantum kernel."""
    def __init__(self, num_qubits: int, depth: int, num_support: int = 20) -> None:
        super().__init__()
        # Quantum kernel
        self.kernel = Kernel()
        self.support_vectors = nn.Parameter(torch.randn(num_support, num_qubits), requires_grad=False)
        # Classical read‑out network
        self.classifier, _, _, _ = build_classifier_circuit(num_support, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for ``x``."""
        k_features = torch.stack([self.kernel(x, sv) for sv in self.support_vectors], dim=1)
        return self.classifier(k_features)

    def fit(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-3, epochs: int = 100) -> None:
        """Trains only the classical read‑out."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            logits = self.forward(X)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

__all__ = ["SharedClassName", "kernel_matrix", "build_classifier_circuit"]
