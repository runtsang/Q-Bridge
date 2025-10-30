"""Hybrid quantum‑classical QCNN implemented with Pennylane.

The ansatz consists of a Z‑feature map, several two‑qubit convolution
layers, and pooling stages that reduce the qubit count.  The final
expectation value is fed into a tiny classical read‑out head.
"""

import pennylane as qml
import torch
from torch import nn
import numpy as np
from typing import Tuple

# Quantum device
dev = qml.device("default.qubit", wires=8)

def conv_circuit(params: np.ndarray, wires: Tuple[int, int]) -> None:
    """Two‑qubit convolution sub‑circuit used in the QCNN."""
    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires[1], wires[0])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires[0], wires[1])
    qml.RY(params[2], wires=wires[1])
    qml.CNOT(wires[1], wires[0])
    qml.RZ(np.pi / 2, wires=wires[0])

def pool_circuit(params: np.ndarray, wires: Tuple[int, int]) -> None:
    """Two‑qubit pooling sub‑circuit that discards one qubit."""
    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires[1], wires[0])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires[0], wires[1])
    qml.RY(params[2], wires=wires[1])

def conv_layer(num_qubits: int, params: np.ndarray, offset: int) -> int:
    """Apply convolution blocks across all adjacent pairs."""
    for i in range(0, num_qubits, 2):
        conv_circuit(params[offset:offset+3], wires=(i, i+1))
        offset += 3
    return offset

def pool_layer(num_qubits: int, params: np.ndarray, offset: int) -> int:
    """Apply pooling blocks to reduce qubit count by half."""
    for i in range(0, num_qubits, 4):
        pool_circuit(params[offset:offset+3], wires=(i, i+1))
        offset += 3
    return offset

@qml.qnode(dev, interface="torch")
def qnn_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Full QCNN ansatz: feature map → convolutions → pooling → measurement."""
    # Feature map (Z‑feature map simplified)
    for i, wire in enumerate(range(dev.nwires)):
        qml.RY(inputs[i], wires=wire)

    # Convolution and pooling stages with weight parameters
    offset = 0
    # Stage 1
    offset = conv_layer(dev.nwires, weights, offset)
    offset = pool_layer(dev.nwires, weights, offset)
    # Stage 2
    offset = conv_layer(dev.nwires // 2, weights, offset)
    offset = pool_layer(dev.nwires // 2, weights, offset)
    # Stage 3
    offset = conv_layer(dev.nwires // 4, weights, offset)
    offset = pool_layer(dev.nwires // 4, weights, offset)

    # Measurement
    return qml.expval(qml.PauliZ(0))

class QCNNHybrid(nn.Module):
    """Hybrid quantum‑classical QCNN with a small classical readout head."""

    def __init__(self, input_dim: int = 8, num_qubits: int = 8) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        # Total number of trainable parameters: 3 per conv/pool block
        num_blocks = 3 * (num_qubits + num_qubits // 2 + num_qubits // 4)
        self.weights = nn.Parameter(torch.randn(num_blocks))
        self.readout = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = qnn_circuit(x, self.weights)
        return torch.sigmoid(self.readout(q_out.unsqueeze(-1)))

def QCNN() -> QCNNHybrid:
    """Construct a ready‑to‑train hybrid QCNN."""
    return QCNNHybrid()

def train(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 20,
) -> Tuple[nn.Module, list]:
    """Training loop for the hybrid QCNN with early stopping."""
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        model.train()
        logits = model(X_train).squeeze()
        loss = criterion(logits, y_train.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val).squeeze()
            val_loss = criterion(val_logits, y_val.float()).item()

        history.append((train_loss, val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history

__all__ = ["QCNNHybrid", "QCNN", "train"]
