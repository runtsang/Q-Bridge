"""Hybrid variational convolutional filter implemented with Pennylane.

The class emulates a quantum convolutional filter that can be trained jointly
with a classical loss.  It accepts a 2‑D array, encodes the pixel values as
rotation angles, applies a parameterised variational layer, and returns the
average probability of measuring |1> over all qubits.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Sequence

class ConvFilter:
    """
    Parameters
    ----------
    kernel_size : int
        Size of the square kernel (number of qubits = kernel_size**2).
    shots : int
        Number of shots to use when sampling from the simulator.
    threshold : float
        Pixel value threshold used for data encoding.
    backend : str
        Pennylane device backend (default "default.qubit").
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 1000,
        threshold: float = 0.5,
        backend: str = "default.qubit",
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.device = qml.device(backend, wires=self.n_qubits, shots=shots)

        # Initialise trainable parameters for the variational circuit
        self.params = pnp.random.uniform(0, 2 * np.pi, size=self.n_qubits, requires_grad=True)

        @qml.qnode(self.device, interface="autograd")
        def circuit(params, data_flat):
            # Data encoding: apply RX(pi) if pixel > threshold
            for i, val in enumerate(data_flat):
                if val > self.threshold:
                    qml.RX(np.pi, wires=i)
            # Variational layer: RY rotations
            for i, p in enumerate(params):
                qml.RY(p, wires=i)
            # Entanglement (linear chain)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measurement: expectation of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, data: Sequence[Sequence[float]]) -> float:
        """
        Execute the variational circuit on a single 2‑D kernel.

        Parameters
        ----------
        data : 2‑D sequence of floats

        Returns
        -------
        float
            Expected value of the observable (PauliZ) after the circuit.
        """
        data_flat = np.asarray(data, dtype=np.float32).reshape(-1)
        result = self.circuit(self.params, data_flat)
        # Convert to probability of |1> from PauliZ expectation
        prob_one = (1 - result) / 2
        return float(prob_one)

    def train_step(self, data: Sequence[Sequence[float]], target: float, lr: float = 0.01) -> float:
        """
        Perform a single gradient‑descent step using the provided target.

        Parameters
        ----------
        data : 2‑D sequence of floats
        target : float
            Desired output value (e.g., from a classical classifier).
        lr : float
            Learning rate.

        Returns
        -------
        float
            Loss value after the step.
        """
        data_flat = np.asarray(data, dtype=np.float32).reshape(-1)
        loss = (self.circuit(self.params, data_flat) - target) ** 2
        grads = qml.grad(self.circuit)(self.params, data_flat)
        self.params -= lr * grads
        return float(loss)

__all__ = ["ConvFilter"]
