import pennylane as qml
import torch
import numpy as np
from torch import nn
from typing import Tuple, List

def make_qnode(num_qubits: int, num_layers: int = 2):
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        qml.templates.AngleEmbedding(x, wires=range(num_qubits))
        for layer in range(num_layers):
            qml.templates.BasicEntanglerLayers(params[layer], wires=range(num_qubits))
        return qml.expval(qml.PauliZ(0))
    return circuit

class QuantumAutoencoder:
    def __init__(self, num_qubits: int, num_layers: int = 2, device: str = "cpu"):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.device = device
        self.circuit = make_qnode(num_qubits, num_layers)
        self.params = torch.randn(num_layers, num_qubits, requires_grad=True, device=device)
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.circuit(self.params, x) for x in batch])
    def train(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        epochs: int = 200,
        lr: float = 0.01,
    ):
        optimizer = torch.optim.Adam([self.params], lr=lr)
        mse = nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.encode(data)
            loss = mse(preds, targets)
            loss.backward()
            optimizer.step()

def QuantumAutoencoderFactory(num_qubits: int, num_layers: int = 2) -> QuantumAutoencoder:
    return QuantumAutoencoder(num_qubits, num_layers)

__all__ = [
    "QuantumAutoencoder",
    "QuantumAutoencoderFactory",
    "make_qnode",
]
