"""Quantum circuit factory and kernel implementation.

The module implements a data‑re‑uploading ansatz with a classical RBF
feature‑map followed by a variational block.  The returned circuit
is compatible with the API of the original ``build_classifier_circuit``.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class RBFAnsatz(tq.QuantumModule):
    """Encodes a vector via a radial‑basis‑function mapping.

    The mapping is implemented as a set of Ry rotations whose angles are
    proportional to ``exp(-γ||x−c||²)`` for a set of centres ``c``.
    """
    def __init__(self,
                 centres: Sequence[torch.Tensor],
                 gamma: float = 1.0) -> None:
        super().__init__()
        self.centres = centres
        self.gamma = gamma

    @tq.static_support
    def forward(self,
                q_device: tq.QuantumDevice,
                x: torch.Tensor) -> None:
        for w, centre in enumerate(self.centres):
            dist = torch.sum((x - centre) ** 2, dim=-1)
            angle = torch.exp(-self.gamma * dist)
            func_name_dict["ry"](q_device, wires=[w], params=angle)

class QuantumCircuitModule(tq.QuantumModule):
    """Variational circuit with data re‑uploading and entangling layers."""
    def __init__(self,
                 n_qubits: int,
                 depth: int,
                 centres: Sequence[torch.Tensor],
                 gamma: float = 1.0) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.rbf = RBFAnsatz(centres, gamma)
        # Variational parameters
        self.params = tq.ParameterList([tq.Parameter() for _ in range(n_qubits * depth)])

    @tq.static_support
    def forward(self,
                q_device: tq.QuantumDevice,
                x: torch.Tensor) -> None:
        # Initial RBF encoding
        self.rbf(q_device, x)
        # Variational re‑uploading
        idx = 0
        for _ in range(self.depth):
            for w in range(self.n_qubits):
                func_name_dict["ry"](q_device, wires=[w], params=self.params[idx])
                idx += 1
            # Entangling layer
            for w in range(self.n_qubits - 1):
                func_name_dict["cz"](q_device, wires=[w, w + 1])

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate the overlap between two encoded states."""
        q_device = tq.QuantumDevice(n_wires=self.n_qubits)
        self.forward(q_device, x)
        # Apply inverse encoding for y
        self.rbf(q_device, -y)
        return torch.abs(q_device.states.view(-1)[0])

def build_classifier_circuit(num_qubits: int,
                             depth: int,
                             centres: Sequence[torch.Tensor] | None = None,
                             gamma: float = 1.0) -> Tuple[QuantumCircuitModule,
                                                          Iterable,
                                                          Iterable,
                                                          list[tq.QuantumDevice]]:
    """Return a quantum classifier circuit and metadata.

    The signature is kept compatible with the original factory.
    ``encoding`` and ``weights`` are lists of parameter names used by
    external optimisation loops.  ``observables`` are placeholder
    ``QuantumDevice`` objects that expose the measurement results.
    """
    if centres is None:
        # default to a grid of random centres
        centres = [torch.randn(1) for _ in range(num_qubits)]
    circuit = QuantumCircuitModule(num_qubits, depth, centres, gamma)
    encoding = list(range(num_qubits))
    weights = list(range(num_qubits * depth))
    observables = [tq.QuantumDevice(n_wires=num_qubits)]
    return circuit, encoding, weights, observables

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix using the quantum kernel."""
    # Use a single circuit instance to reuse parameters
    circuit = QuantumCircuitModule(4, 2, [torch.randn(1) for _ in range(4)])
    return np.array([[circuit.kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumCircuitModule", "build_classifier_circuit", "kernel_matrix"]
