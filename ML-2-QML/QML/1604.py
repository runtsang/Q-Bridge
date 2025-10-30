"""Quantum kernel implementation using PennyLane."""
from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from typing import Sequence, Optional


class HybridKernel:
    """Quantum kernel based on a parameterized ansatz evaluated by fidelity.

    Parameters
    ----------
    wires : int, default=4
        Number of qubits.
    layers : int, default=2
        Depth of the ansatz.
    dev : qml.Device, optional
        PennyLane device. If None, a default.qubit simulator is created.
    """

    def __init__(
        self,
        wires: int = 4,
        layers: int = 2,
        dev: Optional[qml.Device] = None,
    ) -> None:
        self.wires = wires
        self.layers = layers
        self.dev = dev if dev is not None else qml.device("default.qubit", wires=wires)
        self._build_ansatz()

    def _build_ansatz(self) -> None:
        """Create a simple hardware‑efficient ansatz with rotation and entangling gates."""
        @qml.qnode(self.dev, interface="torch")
        def circuit(params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Encode classical data x and y by rotating the first layer
            for i in range(self.wires):
                qml.RY(x[i], wires=i)
            # Apply parameterized layers
            for _ in range(self.layers):
                for i in range(self.wires):
                    qml.RY(params[i], wires=i)
                for i in range(self.wires - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Uncompute y
            for i in range(self.wires):
                qml.RY(-y[i], wires=i)
            return qml.expval(qml.PauliZ(0))  # Arbitrary observable

        self._circuit = circuit

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the quantum kernel value between two 1‑D tensors."""
        # Ensure tensors are 1‑D and match number of wires
        x = x.flatten()
        y = y.flatten()
        if x.shape[0]!= self.wires or y.shape[0]!= self.wires:
            raise ValueError(f"Input vectors must have length {self.wires}")

        # Sample a random parameter vector; in practice this would be optimized
        torch.manual_seed(0)
        params = torch.randn(self.wires, requires_grad=False)

        # Evaluate circuit for both inputs
        val = self._circuit(params, x, y)
        # Kernel value defined as absolute overlap (fidelity approximation)
        return torch.abs(val)

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Compute Gram matrix between two collections of vectors."""
        matrix = torch.stack(
            [
                torch.stack([self.forward(x, y) for y in b])
                for x in a
            ],
            dim=0,
        )
        return matrix.squeeze().cpu().numpy()

__all__ = ["HybridKernel"]
