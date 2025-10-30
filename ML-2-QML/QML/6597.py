import pennylane as qml
import torch
import torch.nn as nn
import pennylane.numpy as np
from pennylane.qnn import CircuitQNN
from typing import Sequence

class QCNNModel(nn.Module):
    """
    Quantum convolutional neural network implemented with PennyLane.
    Wraps a CircuitQNN and exposes a torch.nn.Module interface.
    """

    def __init__(self, num_qubits: int = 8):
        super().__init__()
        self.num_qubits = num_qubits
        self.qnn = self._build_qcnn(num_qubits)

    def _build_qcnn(self, num_qubits: int):
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="autograd")
        def circuit(x, params):
            # Feature map: simple Z‑rotation encoding
            for i, val in enumerate(x):
                qml.RZ(val, wires=i)

            # Convolution and pooling layers
            # Layer 1: convolution on all qubits
            param_index = 0
            for q in range(0, num_qubits, 2):
                idx = param_index
                qml.RZ(-np.pi / 2, wires=q + 1)
                qml.CNOT(wires=[q + 1, q])
                qml.RZ(params[idx], wires=q)
                qml.RY(params[idx + 1], wires=q + 1)
                qml.CNOT(wires=[q, q + 1])
                qml.RY(params[idx + 2], wires=q + 1)
                qml.CNOT(wires=[q + 1, q])
                qml.RZ(np.pi / 2, wires=q)
                param_index += 3

            # Pooling: keep first qubit of each pair (implicit by not using second)

            # Layer 2: convolution on remaining qubits
            remaining = num_qubits // 2
            for q in range(0, remaining, 2):
                idx = param_index
                qml.RZ(-np.pi / 2, wires=q + 1 + remaining)
                qml.CNOT(wires=[q + 1 + remaining, q + remaining])
                qml.RZ(params[idx], wires=q + remaining)
                qml.RY(params[idx + 1], wires=q + 1 + remaining)
                qml.CNOT(wires=[q + remaining, q + 1 + remaining])
                qml.RY(params[idx + 2], wires=q + 1 + remaining)
                qml.CNOT(wires=[q + 1 + remaining, q + remaining])
                qml.RZ(np.pi / 2, wires=q + remaining)
                param_index += 3

            # Pooling: keep first qubit of each pair

            # Layer 3: convolution on last two qubits
            q = 0
            idx = param_index
            qml.RZ(-np.pi / 2, wires=1 + remaining)
            qml.CNOT(wires=[1 + remaining, 0 + remaining])
            qml.RZ(params[idx], wires=0 + remaining)
            qml.RY(params[idx + 1], wires=1 + remaining)
            qml.CNOT(wires=[0 + remaining, 1 + remaining])
            qml.RY(params[idx + 2], wires=1 + remaining)
            qml.CNOT(wires=[1 + remaining, 0 + remaining])
            qml.RZ(np.pi / 2, wires=0 + remaining)

            # Return expectation of Z on first qubit
            return qml.expval(qml.PauliZ(0))

        # Total number of parameters
        total_params = (num_qubits // 2) * 3 + (num_qubits // 4) * 3 + 3
        return CircuitQNN(circuit, interface="autograd",
                          expval_to_prob=lambda v: v,
                          gradient_fn=None,
                          num_weights=total_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        out = self.qnn(x_np)
        return torch.tensor(out, dtype=x.dtype, device=x.device)

def QCNN() -> QCNNModel:
    """Factory returning a quantum :class:`QCNNModel` instance."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
