"""
EstimatorQNN (quantum)

A variational quantum circuit that mirrors the classical model.  The
circuit applies a feature map (RY rotations) followed by a multi‑layer
ansatz of RZ rotations and CNOT entangling gates.  The expectation
value of the Pauli‑Z operator on the first qubit is returned as the
regression prediction.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class EstimatorQNN:
    """
    Variational quantum estimator.

    Parameters
    ----------
    n_qubits : int, default 1
        Number of qubits used in the circuit.
    n_layers : int, default 2
        Number of ansatz layers.
    """

    def __init__(self, n_qubits: int = 1, n_layers: int = 2) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
            # Feature map
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # Ansatz
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RZ(weights[layer, i], wires=i)
                # Entangling layer (linear chain)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def predict(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for each sample.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples, n_qubits) containing input angles.
        params : np.ndarray
            Weight matrix of shape (n_layers, n_qubits).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) with regression predictions.
        """
        return np.array([self.circuit(x, params) for x in X])

__all__ = ["EstimatorQNN"]
