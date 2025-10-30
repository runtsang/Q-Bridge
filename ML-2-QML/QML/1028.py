"""ConvQuantum: Variational quantum convolution implemented with Pennylane.

The class mirrors the classical ConvEnhanced but uses a Pennylane
quantum node to evaluate a tiny variational circuit.  The circuit
consists of a single RY rotation per qubit followed by a full‑chain
CNOT entanglement.  The rotation angles are set to π if the input
pixel exceeds a fixed threshold, otherwise 0.  The output is the
average probability of measuring |1> across all qubits.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import qnode
from pennylane import Device
from pennylane import default_qubit

class ConvQuantum:
    """Drop‑in quantum convolution using Pennylane.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the square kernel (default: 2).
    threshold : float, optional
        Threshold used to encode input pixels into rotation angles
        (default: 127).
    device : pennylane.Device, optional
        Quantum device to run the circuit on.  If None, a
        default_qubit simulator is used.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 127.0, device: Device | None = None) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.device = device or default_qubit.Device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs: np.ndarray) -> np.ndarray:
            # Encode inputs as rotation angles
            for i in range(self.n_qubits):
                angle = np.pi if inputs[i] > self.threshold else 0.0
                qml.RY(angle, wires=i)
            # Full‑chain CNOT entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._circuit = circuit

    def run(self, data: np.ndarray) -> float:
        """
        Run a single patch through the quantum circuit.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flat = data.reshape(-1)
        # Convert expectation values of Z to probabilities of |1>
        expvals = self._circuit(flat)
        probs = (1 - expvals) / 2
        return probs.mean().item()

    def __call__(self, data: np.ndarray) -> float:
        """Convenience alias for ``run``."""
        return self.run(data)

__all__ = ["ConvQuantum"]
