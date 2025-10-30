from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence, Callable

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that implements a simple Ry data‑encoding ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.qd = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.QuantumModule()
        # Build a list of Ry gates for each wire
        self.ansatz.layers = [
            tq.ry(0), tq.ry(1), tq.ry(2), tq.ry(3)
        ]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.qd.reset_states(x.shape[0])
        # encode x
        for i, gate in enumerate(self.ansatz.layers):
            gate(self.qd, wires=[i], params=x[:, i] if gate.num_params else None)
        # inverse encode y
        for i, gate in enumerate(reversed(self.ansatz.layers)):
            gate(
                self.qd,
                wires=[i],
                params=-y[:, i] if gate.num_params else None,
            )
        return torch.abs(self.qd.states.view(-1)[0])

class QuantumFraudCircuit(tq.QuantumModule):
    """Discrete‑gate fraud detection circuit inspired by a two‑mode photonic network."""
    def __init__(self, params: FraudLayerParameters) -> None:
        super().__init__()
        self.params = params
        self.qd = tq.QuantumDevice(n_wires=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.qd.reset_states(x.shape[0])
        # Encode input features into qubits
        for i in range(2):
            tq.rx(self.qd, wires=[i], params=x[:, i] if x.shape[1] > i else None)
        # Entangle the qubits to mimic a beam‑splitter
        tq.cnot(self.qd, wires=[0, 1])
        # Apply phase shifts analogous to the photonic phases
        for i in range(2):
            tq.rz(self.qd, wires=[i], params=self.params.phases[i])
        # Return the real part of the state vector as a measurement proxy
        return self.qd.states.real

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b`` using the quantum kernel."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernel", "QuantumFraudCircuit", "kernel_matrix"]
