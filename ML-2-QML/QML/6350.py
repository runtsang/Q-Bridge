"""Quantum hybrid model with a variational circuit inspired by Quantum‑NAT."""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATHybrid(nn.Module):
    """Quantum model that processes image features through a variational circuit.

    The network consists of:
    - A classical CNN encoder producing a 4‑dimensional feature vector.
    - A 4‑qubit variational circuit parameterized by trainable rotation angles.
    - Measurement of Pauli‑Z on each qubit to yield the final output.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4, dev: str = "default.qubit") -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(8 * 4, 4),  # produce 4‑dimensional vector
        )

        self.dev = qml.device(dev, wires=4)
        # Initialize trainable parameters for the variational circuit
        self.theta = nn.Parameter(torch.randn(4, requires_grad=True))

        self.norm = nn.BatchNorm1d(num_classes)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(params: torch.Tensor):
            for i in range(4):
                qml.RY(params[i], wires=i)
            for i in range(4):
                qml.CNOT(wires=[i, (i + 1) % 4])
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        out = self.circuit(self.theta)
        out = out + encoded
        return self.norm(out)


__all__ = ["QuantumNATHybrid"]
