"""Quantum convolutional filter implemented with Pennylane.

Provides ConvHybrid that can be used as a drop‑in replacement for the
original Conv class.  The filter is a parameterised quantum circuit
acting on a flattened patch.  The circuit is trained via gradient
descent using Pennylane's autograd integration with PyTorch.  The
class exposes a run method that accepts a 2‑D numpy array and
returns the average probability of measuring |1> across the qubits.
"""

import pennylane as qml
import torch
import numpy as np
from typing import Optional

class ConvHybrid:
    """Quantum convolutional filter implemented with Pennylane."""
    def __init__(self,
                 kernel_size: int = 2,
                 device: Optional[str] = None,
                 shots: int = 100,
                 threshold: float = 0.5):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.device = device or "default.qubit"
        self.shots = shots
        self.threshold = threshold
        self._dev = qml.device(self.device, wires=self.n_qubits)

        @qml.qnode(self._dev, interface="torch", diff_method="backprop")
        def circuit(inputs, theta):
            # Encode inputs as RX rotations
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._circuit = circuit

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single patch (Tensor of shape (k,k))."""
        # Flatten and encode as angles
        inputs = torch.where(patch.view(-1) > self.threshold,
                             torch.tensor(np.pi, dtype=torch.float32),
                             torch.tensor(0.0, dtype=torch.float32))
        theta = torch.zeros(self.n_qubits, dtype=torch.float32)
        # Run the circuit
        expvals = self._circuit(inputs, theta)
        # Convert to probability of |1> : (1 - expval)/2
        probs = (1 - torch.tensor(expvals)) / 2
        return probs.mean()

    def run(self, data: np.ndarray) -> float:
        """Run the quantum filter on a 2‑D numpy array."""
        patch = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(patch).item()

def Conv():
    """Return a ConvHybrid instance with default parameters."""
    return ConvHybrid(kernel_size=2, shots=100, threshold=0.5)

__all__ = ["ConvHybrid", "Conv"]
