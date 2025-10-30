"""Quantum estimator with a multi‑qubit variational circuit and entangling layers."""

import pennylane as qml
import numpy as np

class EstimatorQNNExtended:
    """
    Variational quantum circuit estimator.

    Parameters
    ----------
    num_qubits : int, default 2
        Number of qubits in the circuit.
    layers : int, default 2
        Number of variational layers.
    """

    def __init__(self, num_qubits: int = 2, layers: int = 2) -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        # Device capable of simulating the circuit
        self.dev = qml.device("default.qubit", wires=num_qubits)
        # Define the QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")
        # Observables: expectation of Pauli‑Z on each qubit
        self.observables = [qml.PauliZ(i) for i in range(num_qubits)]

    def _circuit(self, inputs: np.ndarray, weights: np.ndarray) -> list[float]:
        """
        Build the variational circuit.

        Parameters
        ----------
        inputs : np.ndarray
            Input features of shape (num_qubits,).
        weights : np.ndarray
            Weight parameters of shape (layers, num_qubits).

        Returns
        -------
        list[float]
            Expectation values for each observable.
        """
        # Data encoding: RX gates
        for i, wire in enumerate(range(self.num_qubits)):
            qml.RX(inputs[i], wires=wire)
        # Variational layers
        for layer in range(self.layers):
            for i in range(self.num_qubits):
                qml.RY(weights[layer, i], wires=i)
            # Entangling pattern: linear chain
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(obs) for obs in self.observables]

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a single input sample.

        Parameters
        ----------
        inputs : np.ndarray
            Input vector of shape (num_qubits,).

        Returns
        -------
        np.ndarray
            Array of expectation values for each observable.
        """
        # Random initial weights; in practice these would be trained
        rng = np.random.default_rng()
        weights = rng.uniform(-np.pi, np.pi, size=(self.layers, self.num_qubits))
        return self.qnode(inputs, weights)

def EstimatorQNN() -> EstimatorQNNExtended:
    """
    Helper that returns the default EstimatorQNNExtended instance
    compatible with the original EstimatorQNN interface.
    """
    return EstimatorQNNExtended()

__all__ = ["EstimatorQNNExtended", "EstimatorQNN"]
