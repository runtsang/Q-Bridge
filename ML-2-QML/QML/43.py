"""Variational QCNN implemented with PennyLane, featuring a parameterized ansatz and adaptive pooling."""

import pennylane as qml
import torch
import numpy as np

class QCNNHybrid:
    """
    Quantum convolutional neural network using PennyLane with a Torch interface.
    The circuit implements the classic QCNN layout:
    - Z‑feature map
    - Three convolution layers (2‑qubit entangling blocks)
    - Three pooling layers (2‑qubit reduction blocks)
    The class exposes a simple training API and integrates seamlessly with
    classical PyTorch models.
    """
    def __init__(self,
                 n_qubits: int = 8,
                 device: str = "default.qubit",
                 shots: int = 1024,
                 seed: int | None = 12345):
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits, shots=shots, seed=seed)

        # Total number of trainable parameters in the ansatz
        self._num_params = 42  # 12+12+6+6+3+3
        # Parameter vector used by the QNode
        self.params = torch.nn.Parameter(torch.randn(self._num_params))

        # QNode with Torch interface
        self.qnode = qml.QNode(self._qcircuit, self.dev, interface="torch")

        # Optimizer
        self.optimizer = torch.optim.Adam([self.params], lr=0.01)

    # ------------------------------------------------------------------
    # Convolution and pooling primitives
    # ------------------------------------------------------------------
    def _conv_circuit(self, params, wires):
        """Two‑qubit convolution unit (parameterized by 3 angles)."""
        qml.RZ(-np.pi / 2, wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(params[2], wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RZ(np.pi / 2, wires=wires[0])

    def _pool_circuit(self, params, wires):
        """Two‑qubit pooling unit (parameterized by 3 angles)."""
        qml.RZ(-np.pi / 2, wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(params[2], wires=wires[1])

    # ------------------------------------------------------------------
    # Core circuit
    # ------------------------------------------------------------------
    def _qcircuit(self, features: torch.Tensor, params: torch.Tensor):
        """
        Full QCNN circuit combining the feature map and ansatz.
        `features` is a vector of size `n_qubits`; `params` is a flat vector of length 42.
        """
        # Feature map: simple Z‑RX rotation per qubit
        for i in range(self.n_qubits):
            qml.RZ(features[i], wires=i)
            qml.RX(features[i], wires=i)

        # Layer 1: 4 convolutions (pairs 0‑1, 2‑3, 4‑5, 6‑7)
        idx = 0
        for i in range(0, 8, 2):
            self._conv_circuit(params[idx:idx+3], wires=[i, i+1])
            idx += 3

        # Layer 1: 4 poolings (pairs 0‑1, 2‑3, 4‑5, 6‑7)
        for i in range(0, 8, 2):
            self._pool_circuit(params[idx:idx+3], wires=[i, i+1])
            idx += 3

        # Layer 2: 2 convolutions (pairs 4‑5, 6‑7)
        for i in range(4, 8, 2):
            self._conv_circuit(params[idx:idx+3], wires=[i, i+1])
            idx += 3

        # Layer 2: 2 poolings (pairs 4‑5, 6‑7)
        for i in range(4, 8, 2):
            self._pool_circuit(params[idx:idx+3], wires=[i, i+1])
            idx += 3

        # Layer 3: 1 convolution (pair 6‑7)
        self._conv_circuit(params[idx:idx+3], wires=[6, 7])
        idx += 3

        # Layer 3: 1 pooling (pair 6‑7)
        self._pool_circuit(params[idx:idx+3], wires=[6, 7])

        # Measurement: expectation of Z on qubit 0
        return qml.expval(qml.PauliZ(0))

    # ------------------------------------------------------------------
    # PyTorch integration
    # ------------------------------------------------------------------
    def forward(self, features: torch.Tensor):
        """
        Compute the expectation value for a batch of feature vectors.
        `features` should be of shape (batch, n_qubits).
        """
        return self.qnode(features, self.params)

    def train_step(self, features: torch.Tensor,
                   labels: torch.Tensor,
                   loss_fn: torch.nn.Module):
        """
        Perform one gradient‑descent step.
        """
        self.optimizer.zero_grad()
        preds = self.forward(features)
        loss = loss_fn(preds, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, features: torch.Tensor, threshold: float = 0.5):
        """
        Return binary predictions for a batch of inputs.
        """
        with torch.no_grad():
            probs = self.forward(features)
        return (probs >= threshold).float()

__all__ = ["QCNNHybrid"]
