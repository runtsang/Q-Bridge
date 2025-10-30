"""Quantum neural network estimator using Pennylane.

The circuit implements a two‑qubit variational ansatz with
parameterised rotations and entangling gates.  The output is the
expectation value of the Pauli‑Y operator on qubit 0, which is
used as the regression target.
"""

import pennylane as qml
import pennylane.numpy as np
from typing import Sequence

class EstimatorQNN:
    """Variational quantum circuit for regression.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    layers : int
        Number of variational layers.
    seed : int | None
        Random seed for weight initialization.
    """
    def __init__(self, n_qubits: int = 2, layers: int = 2, seed: int | None = None) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        if seed is not None:
            np.random.seed(seed)
        # Initialise weights with small random values
        self.weight_params = 0.01 * np.random.randn(layers, n_qubits, 3)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: Sequence[float], weights: np.ndarray) -> float:
            # Encode inputs using RY rotations
            for i, w in enumerate(inputs):
                qml.RY(w, wires=i)
            # Variational layers
            for l in range(layers):
                for q in range(n_qubits):
                    qml.Rot(*weights[l, q, :], wires=q)
                # Entangling layer (ring topology)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            # Measurement
            return qml.expval(qml.PauliY(0))

        self.circuit = circuit

    def __call__(self, inputs: Sequence[float]) -> float:
        """Evaluate the circuit with current weights."""
        return float(self.circuit(inputs, self.weight_params))

    def parameters(self):
        """Return the trainable parameters."""
        return self.weight_params

    def set_parameters(self, params: np.ndarray) -> None:
        """Set the trainable parameters."""
        self.weight_params = params

__all__ = ["EstimatorQNN"]
