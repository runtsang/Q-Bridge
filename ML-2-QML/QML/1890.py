"""Hybrid classical-quantum binary classifier – quantum branch.

This module implements the same public interface as the classical
counterpart but replaces the dense head with a variational quantum
circuit built with Pennylane.  The circuit supports a feature map,
parameter‑shift gradient estimation, and an arbitrary number of qubits
and layers."""
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

class QuantumNode(nn.Module):
    """Pennylane quantum node that maps a single scalar to a
    single expectation value."""
    def __init__(self,
                 n_qubits: int = 2,
                 n_layers: int = 2,
                 shift: float = np.pi / 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.shift = shift
        dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, dev, interface="torch")

        # Initialise variational weights
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

    def _circuit(self, x, weights):
        # Feature map – encode input into rotation about Z
        qml.RZ(x + self.shift, wires=0)
        # Variational layers
        for layer in weights:
            for i, w in enumerate(layer):
                qml.RX(w[0], wires=i)
                qml.RY(w[1], wires=i)
                qml.RZ(w[2], wires=i)
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expectation value returned as a 1‑D tensor
        return self.qnode(x, self.weights)

class HybridBinaryClassifier(nn.Module):
    """ResNet18 backbone with a quantum head.  The quantum head
    replaces the final dense layer of the classical model and
    provides a true quantum contribution to the predictions."""
    def __init__(self,
                 pretrained: bool = True,
                 n_qubits: int = 2,
                 n_layers: int = 2,
                 shift: float = np.pi / 2):
        super().__init__()
        # Backbone identical to the classical side
        self.backbone = models.resnet18(pretrained=pretrained)
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False
        self.backbone.fc = nn.Identity()

        # Quantum head
        self.quantum_head = QuantumNode(n_qubits=n_qubits,
                                        n_layers=n_layers,
                                        shift=shift)

        # Final linear layer to map quantum expectation to logits
        self.logit = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # Reduce features to a single scalar before feeding to quantum circuit
        scalar = torch.mean(features, dim=1, keepdim=True)
        q_expect = self.quantum_head(scalar)
        logits = self.logit(q_expect)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier", "QuantumNode"]
