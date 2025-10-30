"""ConvGen275Q: Quantum variational filter inspired by ConvGen275."""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class ConvGen275Q:
    """
    Variational quantum filter that mirrors the classical ConvGen275.
    Encodes a 2×2 patch (or larger) into qubit states, applies a parameterized
    RX rotation per qubit, entangles with a CX ladder, and measures qubits.
    The parameters are learnable and can be optimized with a quantum‑classical
    hybrid loop.

    The class exposes a `run` method that accepts a 2‑D array and returns the
    average probability of measuring |1> across all qubits.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 1024,
        threshold: float = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size * kernel_size
        self.shots = shots
        self.threshold = threshold

        # PennyLane device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Initialize parameters
        self.params = pnp.random.uniform(0, 2 * np.pi, self.n_qubits, requires_grad=True)

        # Build circuit
        @qml.qnode(self.dev, interface="autograd")
        def circuit(data, params):
            # Angle encoding
            for i, val in enumerate(data):
                qml.RY(val, wires=i)
            # Parameterized rotations
            for i in range(self.n_qubits):
                qml.RX(params[i], wires=i)
            # Entanglement (CX ladder)
            for i in range(self.n_qubits - 1):
                qml.CX(wires=[i, i + 1])
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data) -> float:
        """
        Execute the circuit on a single 2‑D patch.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size) with values
                  in [0, 1] (e.g., normalized pixel intensities).

        Returns:
            Mean probability of measuring |1> across all qubits.
        """
        # Flatten and normalize data to [0, π]
        flat = np.array(data).reshape(-1)
        flat = np.clip(flat, 0, 1) * np.pi
        # Evaluate circuit
        expectation = self.circuit(flat, self.params)
        # Convert expval of PauliZ to probability of |1>
        probs = (1 - expectation) / 2
        return probs.mean().item()

    def get_params(self):
        """Return current parameters."""
        return self.params

    def set_params(self, new_params):
        """Set new parameters."""
        self.params = new_params

__all__ = ["ConvGen275Q"]
