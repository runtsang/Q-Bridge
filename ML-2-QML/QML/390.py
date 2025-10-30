"""ConvGen: Quantum‑classical convolutional filter using Pennylane.

The interface mirrors the classical version: a ``Conv()`` function
returns an object with a ``run`` method.  The circuit uses a
parameter‑shiftable variational ansatz so that the angles are
trainable with a quantum‑classical hybrid optimiser.  The filter
produces a scalar that is the mean probability of measuring |1>
across all qubits, matching the seed’s behaviour.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

__all__ = ["Conv"]


def Conv():
    class QuanvCircuit:
        """Filter circuit used for quanvolution layers."""

        def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 127):
            self.n_qubits = kernel_size ** 2
            self.shots = shots
            self.threshold = threshold

            # Create a parameter‑shiftable device
            self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)

            # Initialise trainable parameters
            self.params = np.random.uniform(0, 2 * np.pi, size=self.n_qubits)

            # Build a simple variational ansatz
            @qml.qnode(self.dev, interface="autograd", diff_method="parameter-shift")
            def circuit(theta, data):
                # Encode data by rotating each qubit
                for i, val in enumerate(data):
                    angle = np.pi if val > self.threshold else 0.0
                    qml.RY(angle, wires=i)
                # Variational layer
                for i in range(self.n_qubits):
                    qml.RY(theta[i], wires=i)
                # Entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Measurement
                return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

            self.circuit = circuit

        def run(self, data):
            """Run the quantum circuit on classical data.

            Args:
                data: 2D array with shape (kernel_size, kernel_size).

            Returns:
                float: average probability of measuring |1> across qubits.
            """
            flat = np.reshape(data, (self.n_qubits,))
            # Prepare data for encoding
            result = self.circuit(self.params, flat)
            # Convert expectation values of Z to probabilities of |1>
            probs = (1 - np.array(result)) / 2
            return probs.mean().item()

        def trainable_params(self):
            """Return the trainable parameters for optimisation."""
            return self.params

        def set_params(self, new_params):
            """Set new parameters for the variational layer."""
            self.params = np.array(new_params)

    return QuanvCircuit()
