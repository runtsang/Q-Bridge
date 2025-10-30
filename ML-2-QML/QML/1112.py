"""
Quantum convolutional neural network implemented with Pennylane.
Provides a variational circuit with convolution and pooling layers
and a gradient‑based training loop that integrates with PyTorch.
"""

from __future__ import annotations

import torch
import pennylane as qml
import pennylane.numpy as np
from pennylane import qnn
from typing import Iterable, Tuple


class QCNNModel:
    """
    Variational QCNN that maps 8‑dimensional classical inputs to a
    single binary output.  The circuit consists of a Z‑feature map,
    three convolutional layers followed by adaptive pooling, and a
    single Pauli‑Z measurement.  Parameters are optimised with
    Adam via a TorchLayer wrapper.
    """
    def __init__(self, n_qubits: int = 8, seed: int = 42) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        np.random.seed(seed)

        # Trainable parameters for each convolution and pooling block
        self.conv_params = np.random.uniform(0, 2 * np.pi, (n_qubits // 2, 3))
        self.pool_params = np.random.uniform(0, 2 * np.pi, (n_qubits // 4, 3))

        # Feature map
        self.feature_map = qml.templates.feature_maps.ZFeatureMap(n_qubits)

        # Build the quantum neural network
        self._build_qnn()

    def _build_qnn(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, conv: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
            # Encode data
            self.feature_map(inputs)

            # Convolution layers (3 layers, each acting on two qubits)
            for i in range(0, self.n_qubits, 2):
                idx = i // 2
                qml.RZ(conv[idx, 0], wires=i)
                qml.RY(conv[idx, 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(conv[idx, 2], wires=i)
                qml.RY(conv[idx, 0], wires=i + 1)
                qml.CNOT(wires=[i + 1, i])

            # Adaptive pooling layer (one block per pair of qubits)
            for i in range(0, self.n_qubits, 4):
                idx = i // 4
                qml.RZ(pool[idx, 0], wires=i)
                qml.RY(pool[idx, 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(pool[idx, 2], wires=i)

            # Output expectation value
            return qml.expval(qml.PauliZ(0))

        # Wrap the qnode in a TorchLayer for automatic differentiation
        self.qnn = qnn.TorchLayer(circuit, output_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the variational circuit.
        """
        return self.qnn(x, self.conv_params, self.pool_params)

    def loss(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Binary cross‑entropy loss.
        """
        return torch.nn.functional.binary_cross_entropy(preds, labels.unsqueeze(1))

    def train_loop(
        self,
        dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 100,
        lr: float = 0.01,
        device: str = "cpu",
    ) -> None:
        """
        Gradient‑based training loop using Adam optimiser.
        """
        self.conv_params = torch.tensor(self.conv_params, requires_grad=True, device=device)
        self.pool_params = torch.tensor(self.pool_params, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([self.conv_params, self.pool_params], lr=lr)
        self.qnn.to(device)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = self.forward(batch_x)
                loss = self.loss(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss/len(dataloader):.4f}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return binary predictions (0 or 1) for the given inputs.
        """
        with torch.no_grad():
            probs = self.forward(x)
            return (probs > 0.5).float()


def QCNN() -> QCNNModel:
    """
    Factory returning a ready‑to‑train QCNNModel instance.
    """
    return QCNNModel()


__all__ = ["QCNNModel", "QCNN"]
