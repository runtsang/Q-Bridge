"""Quantum version of the hybrid binary classifier using Pennylane."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

import pennylane as qml


class HybridQuantumNet(nn.Module):
    """
    Hybrid quantum‑classical binary classifier.

    Architecture:
      * ResNet‑18 backbone for feature extraction.
      * A linear head producing a scalar logit.
      * A four‑qubit variational circuit whose expectation value
        on the first qubit acts as the quantum head.
    """

    def __init__(self, n_qubits: int = 4, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.backbone = resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, 1)

        self.n_qubits = n_qubits
        self.shift = shift
        self.qml_device = qml.device("default.qubit", wires=self.n_qubits)

        # Pre‑define the variational circuit as a QNode
        @qml.qnode(self.qml_device, interface="torch", diff_method="parameter-shift")
        def circuit(theta: torch.Tensor) -> torch.Tensor:
            for i in range(self.n_qubits):
                qml.RY(theta[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.backbone(inputs)
        logits = self.fc(features)

        # Map the scalar logit to a parameter vector for the circuit
        theta = torch.sigmoid(logits).view(-1, 1).repeat(1, self.n_qubits)

        # Quantum expectation value
        q_expect = self.circuit(theta)

        probs = torch.sigmoid(q_expect + self.shift)
        return torch.cat((probs, 1 - probs), dim=1)


__all__ = ["HybridQuantumNet"]
