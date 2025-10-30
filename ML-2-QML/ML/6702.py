"""Hybrid classical‑quantum convolutional integrator.

This module extends the original Conv filter by adding a
parameter‑ised quantum circuit that can be trained jointly
with the classical convolutional weights.  It keeps the same
factory interface – calling ``ConvIntegrator()`` returns an
object with a ``run`` method that accepts a 2‑D NumPy array
and returns a scalar.  Internally the class is a PyTorch
``nn.Module`` so the quantum parameters are registered as
``torch.nn.Parameter``s and can be optimized with standard
optimizers.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import pennylane as qml
from pennylane import qiskit as qiskit_plugin

class ConvIntegrator(nn.Module):
    """
    Hybrid convolutional filter that combines a classical 2‑D
    convolution with a small variational quantum circuit.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        quantum_depth: int = 2,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Classical part – a single‑channel 2‑D convolution
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Quantum part – PennyLane device on top of Qiskit Aer
        n_qubits = kernel_size * kernel_size
        self.device = qml.device(
            "qiskit.aer_simulator", wires=n_qubits, shots=1000, device=qiskit_plugin
        )
        self.qubits = n_qubits
        self.quantum_depth = quantum_depth

        # Learnable rotation parameters for each qubit
        self.theta = nn.Parameter(
            torch.randn(n_qubits, dtype=torch.float32)
        )

        # Optional learnable scaling between classical and quantum outputs
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def quantum_circuit(self, x: torch.Tensor) -> torch.Tensor:
        """Parameterized quantum circuit used in the hybrid filter."""
        @qml.qnode(self.device, interface="torch")
        def circuit(params: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
            # Encode the data as rotations on each qubit
            for i in range(self.qubits):
                qml.RX(data[i], wires=i)

            # Variational layers
            for _ in range(self.quantum_depth):
                for i in range(self.qubits):
                    qml.RX(params[i], wires=i)
                # Entangling layer
                for i in range(self.qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Measure expectation of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))

        # Flatten image to 1D tensor
        data = x.view(-1)
        return circuit(self.theta, data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of images.
        x: Tensor of shape (B, 1, H, W)
        Returns a scalar tensor per example (batch size B).
        """
        # Classical convolution
        conv_out = self.conv(x)
        # Global average pooling to get scalar per example
        conv_scalar = conv_out.mean(dim=[2, 3])

        # Quantum output per example
        quantum_scalar = torch.stack(
            [self.quantum_circuit(img.squeeze(0)) for img in x]
        )

        # Combine linearly
        return self.alpha * conv_scalar + (1 - self.alpha) * quantum_scalar

    def run(self, data: np.ndarray) -> float:
        """
        Convenience method matching the original API.
        Accepts a 2‑D NumPy array of shape (kernel_size, kernel_size)
        and returns a scalar float.
        """
        # Convert to torch tensor
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.forward(tensor)
        return output.item()

__all__ = ["ConvIntegrator"]
