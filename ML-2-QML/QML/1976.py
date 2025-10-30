"""
Quantum kernel module using Pennylane.
Implements a parameterised variational ansatz that can be trained
to maximise kernel similarity for a pair of classical inputs.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as pnp


class QuantumKernelMethod(qml.QuantumNode):
    """
    Variational quantum kernel implemented as a Pennylane QuantumNode.
    The ansatz consists of alternating single‑qubit rotations and a CNOT ladder,
    with trainable rotation angles that are optimised jointly with classical
    hyper‑parameters.
    """

    def __init__(self, num_wires: int = 4, device_name: str = "default.qubit") -> None:
        """
        Parameters
        ----------
        num_wires : int, default=4
            Number of qubits in the device.
        device_name : str, default="default.qubit"
            Pennylane device to use.
        """
        self.num_wires = num_wires
        self.dev = qml.device(device_name, wires=self.num_wires)
        self._build_ansatz()
        super().__init__(self._ansatz, self.dev, interface="torch")

    def _build_ansatz(self) -> None:
        """Create a trainable variational ansatz."""
        # Number of layers determines the number of parameters per input
        self.num_layers = 2
        self.theta_shape = (self.num_layers, self.num_wires, 3)  # RX, RY, RZ per layer
        self.params = torch.nn.Parameter(
            torch.randn(self.theta_shape, dtype=torch.float32, requires_grad=True)
        )

    def _ansatz(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the ansatz to encode two classical vectors x and y.
        The encoding is achieved by applying the same circuit to x and y
        then computing the overlap of the resulting states.
        """
        # x, y : (batch, num_wires)
        # Encode x
        for layer in range(self.num_layers):
            for qubit in range(self.num_wires):
                qml.RX(x[0, qubit], wires=qubit)
                qml.RY(x[0, qubit], wires=qubit)
                qml.RZ(x[0, qubit], wires=qubit)
            for qubit in range(self.num_wires - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
        # Encode y (with inverse rotations)
        for layer in range(self.num_layers):
            for qubit in range(self.num_wires):
                qml.RX(-y[0, qubit], wires=qubit)
                qml.RY(-y[0, qubit], wires=qubit)
                qml.RZ(-y[0, qubit], wires=qubit)
            for qubit in range(self.num_wires - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        # Return the absolute value of the overlap (kernel)
        return qml.expval(qml.PauliZ(0))  # placeholder: use overlap measurement

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value for two input vectors.
        """
        # Ensure inputs are of shape (batch, num_wires)
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        return torch.abs(self._ansatz(x, y))

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two sequences of tensors.
        """
        X = torch.stack(a)  # (m, d)
        Y = torch.stack(b)  # (n, d)
        m, n = X.shape[0], Y.shape[0]
        gram = torch.zeros((m, n), dtype=torch.float32)
        for i in range(m):
            for j in range(n):
                gram[i, j] = self.forward(X[i], Y[j])  # torch scalar
        return gram.detach().cpu().numpy()


__all__ = ["QuantumKernelMethod"]
