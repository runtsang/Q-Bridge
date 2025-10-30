"""Quantum QCNN implementation using PennyLane.

The quantum model preserves the core idea of a QCNN:
a feature map followed by a stack of convolutional layers.
Each convolution is a two‑qubit entangling block that
uses parameterised RZ, RX, and CNOT gates.  The network
is fully variational and can be trained with a PyTorch
optimiser, making it a true hybrid classical‑quantum
model.

Author: gpt-oss-20b
"""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
from typing import Iterable


class QCNNModel(nn.Module):
    """Variational QCNN using PennyLane."""

    def __init__(
        self,
        n_qubits: int = 8,
        feature_wires: Iterable[int] | None = None,
        conv_params_per_pair: int = 4,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.n_qubits = n_qubits
        self.feature_wires = list(range(n_qubits)) if feature_wires is None else list(feature_wires)
        self.conv_params_per_pair = conv_params_per_pair

        # Device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Total number of trainable parameters
        n_pairs = n_qubits // 2
        n_conv_layers = 3
        total_params = n_pairs * n_conv_layers * conv_params_per_pair
        self.params = nn.Parameter(torch.randn(total_params))

        # QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            # Feature map: simple RY rotations
            for i, wire in enumerate(self.feature_wires):
                qml.RY(x[i], wires=wire)

            # Helper to apply a conv block on a pair
            def conv_block(pair: tuple[int, int], params: torch.Tensor):
                i, j = pair
                qml.RZ(params[0], wires=i)
                qml.RX(params[1], wires=j)
                qml.CNOT(i, j)
                qml.RZ(params[2], wires=j)
                qml.RX(params[3], wires=j)

            # Convolution layers
            idx = 0
            for layer in range(n_conv_layers):
                for pair_idx in range(n_pairs):
                    pair = (pair_idx * 2, pair_idx * 2 + 1)
                    pair_params = w[idx : idx + self.conv_params_per_pair]
                    conv_block(pair, pair_params)
                    idx += self.conv_params_per_pair

            # Readout
            return qml.expval(qml.PauliZ(self.feature_wires[0]))

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return the QCNN output for a batch of inputs."""
        # Flatten input to match feature map shape
        batch_shape = inputs.shape[:-1]
        flat_inputs = inputs.view(-1, self.n_qubits)
        outputs = []
        for x in flat_inputs:
            outputs.append(self.circuit(x, self.params))
        out = torch.stack(outputs).reshape(batch_shape)
        return torch.sigmoid(out)

    def train_qcnn(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> None:
        """Simple training loop using Adam optimiser."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.forward(X_train).squeeze()
            loss = loss_fn(preds, y_train)
            loss.backward()
            optimizer.step()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return binary predictions (0 or 1)."""
        probs = self.forward(X).detach()
        return (probs > 0.5).long()


def QCNN_QNN() -> QCNNModel:
    """Factory returning a fully‑configured quantum QCNN."""
    return QCNNModel()


__all__ = ["QCNN_QNN", "QCNNModel"]
