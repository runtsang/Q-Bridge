"""Quantum kernel module using PennyLane with a trainable variational ansatz."""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from torch import nn
from typing import Sequence, Dict, Any

class QuantumKernelMethod(nn.Module):
    """
    Quantum kernel that evaluates the squared overlap between
    parameterised quantum states |ψ(x)⟩ and |ψ(y)⟩.

    Parameters
    ----------
    n_wires : int
        Number of qubits in the circuit.
    n_layers : int
        Depth of the variational encoding.
    dev_name : str
        PennyLane device name (e.g. "default.qubit", "qiskit.ibmq_qasm_simulator").
    device_kwargs : dict
        Additional keyword arguments for the PennyLane device.
    """
    def __init__(
        self,
        n_wires: int = 4,
        n_layers: int = 2,
        dev_name: str = "default.qubit",
        device_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev_name = dev_name
        self.device_kwargs = device_kwargs or {}
        self.device = qml.device(self.dev_name, wires=self.n_wires, **self.device_kwargs)

        # Trainable parameters for the variational encoding
        self.params = nn.Parameter(torch.randn(n_wires * n_layers))

        # Compile a QNode that returns the statevector
        @qml.qnode(self.device, interface="torch")
        def _state(x: torch.Tensor):
            # Data encoding: RX rotations
            for i, val in enumerate(x):
                qml.RX(val, wires=i)
            # Variational layers
            for l in range(self.n_layers):
                for w in range(self.n_wires):
                    qml.RY(self.params[l * self.n_wires + w], wires=w)
                for w in range(self.n_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
            return qml.state()

        self._state = _state

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value k(x, y) = |⟨ψ(x)|ψ(y)⟩|².
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        # Batch evaluation – for simplicity we evaluate pairwise one‑by‑one
        k = []
        for xi in x:
            psi_x = self._state(xi)
            for yi in y:
                psi_y = self._state(yi)
                overlap = torch.dot(psi_x.conj(), psi_y)
                k.append(overlap.abs().pow(2))
        return torch.stack(k).reshape(x.shape[0], y.shape[0])

    def kernel_matrix(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two datasets X and Y.
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y if Y is not None else X, dtype=torch.float32)
        K_t = self.forward(X_t, Y_t)
        return K_t.detach().cpu().numpy()

__all__ = ["QuantumKernelMethod"]
