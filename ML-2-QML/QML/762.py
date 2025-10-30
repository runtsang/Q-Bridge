"""Quantum QCNN implemented with Pennylane and PyTorch integration."""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
from typing import Callable, Tuple


class QCNNQuantum(nn.Module):
    """
    Hybrid QCNN that builds a convolution‑pooling quantum circuit in Pennylane.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the feature map (must be even).
    device : str
        Pennylane device name (e.g., "default.qubit", "qiskit.ibmq_qasm_simulator").
    """

    def __init__(self, num_qubits: int = 8, device: str = "default.qubit") -> None:
        super().__init__()
        if num_qubits % 2!= 0:
            raise ValueError("num_qubits must be even for pairwise convolutions.")
        self.num_qubits = num_qubits
        self.dev = qml.device(device, wires=num_qubits)
        self.feature_map = self._build_feature_map()
        self.ansatz = self._build_ansatz()
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _build_feature_map(self) -> Callable:
        """Z‑feature map used to encode classical data."""
        return lambda x: qml.feature_map.ZFeatureMap(x, wires=range(self.num_qubits))

    def _conv_layer(self, params: torch.Tensor, start: int = 0) -> None:
        """Apply a two‑qubit convolution to adjacent qubit pairs."""
        for i in range(0, self.num_qubits, 2):
            p = params[start + i : start + i + 3]
            qml.RZ(-qml.pi / 2, wires=i + 1)
            qml.CNOT(wires=[i + 1, i])
            qml.RZ(p[0], wires=i)
            qml.RY(p[1], wires=i + 1)
            qml.CNOT(wires=[i, i + 1])
            qml.RY(p[2], wires=i + 1)
            qml.CNOT(wires=[i + 1, i])
            qml.RZ(qml.pi / 2, wires=i)
        # No return; operations are applied in‑place

    def _pool_layer(self, params: torch.Tensor, start: int = 0) -> None:
        """Apply a two‑qubit pooling to adjacent qubit pairs."""
        for i in range(0, self.num_qubits, 2):
            p = params[start + i : start + i + 3]
            qml.RZ(-qml.pi / 2, wires=i + 1)
            qml.CNOT(wires=[i + 1, i])
            qml.RZ(p[0], wires=i)
            qml.RY(p[1], wires=i + 1)
            qml.CNOT(wires=[i, i + 1])
            qml.RY(p[2], wires=i + 1)
        # No return; operations are applied in‑place

    def _build_ansatz(self) -> Callable:
        """Construct a layered ansatz with conv‑pool‑conv‑pool‑conv‑pool."""
        def ansatz(params: torch.Tensor) -> None:
            # params shape: (num_layers * num_qubits * 3,)
            layer_params = params.reshape(-1, self.num_qubits, 3)
            # Layer 1: conv
            self._conv_layer(layer_params[0])
            # Layer 1: pool
            self._pool_layer(layer_params[1])
            # Layer 2: conv (on half qubits)
            self._conv_layer(layer_params[2], start=0)
            # Layer 2: pool (on half qubits)
            self._pool_layer(layer_params[3], start=0)
            # Layer 3: conv (on last two qubits)
            self._conv_layer(layer_params[4], start=self.num_qubits - 2)
            # Layer 3: pool (on last two qubits)
            self._pool_layer(layer_params[5], start=self.num_qubits - 2)
        return ansatz

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Full QCNN circuit: feature map → ansatz → measurement."""
        # Encode data
        self.feature_map(x)
        # Apply ansatz
        self.ansatz(params)
        # Measure expectation of Z on the first qubit
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a probability via sigmoid of the expectation value.
        """
        # Randomly initialise parameters if not already set
        if not hasattr(self, "_params"):
            self._params = nn.Parameter(torch.randn(6 * self.num_qubits * 3))
        probs = torch.sigmoid(self.qnode(x, self._params))
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return binary predictions (0 or 1) for a batch of inputs.
        """
        with torch.no_grad():
            probs = self(x)
        return (probs > 0.5).long()


def QCNNQuantumFactory(num_qubits: int = 8, device: str = "default.qubit") -> QCNNQuantum:
    """Convenience factory for the QCNNQuantum model."""
    return QCNNQuantum(num_qubits=num_qubits, device=device)


__all__ = ["QCNNQuantum", "QCNNQuantumFactory"]
