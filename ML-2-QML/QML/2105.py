"""Quantum neural network for regression.

This module builds a hybrid quantum‑classical estimator that augments
the original single‑qubit example with a 3‑qubit entangled circuit
and a weighted observable.  The circuit is parameterised by
input features and trainable weights and returns the expectation
value of a Pauli‑Y operator as the prediction.
"""

import pennylane as qml
import pennylane.numpy as np

class EstimatorQNN:
    """
    Quantum neural network with 3 entangled qubits.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector (default 2).
    weight_dim : int
        Number of trainable weight parameters (default 4).
    """

    def __init__(self, input_dim: int = 2, weight_dim: int = 4) -> None:
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.dev = qml.device("default.qubit", wires=3)

        # Define parameter symbols
        self.input_params = [f"inp_{i}" for i in range(input_dim)]
        self.weight_params = [f"w_{i}" for i in range(weight_dim)]

    def circuit(self, *params):
        """
        Variational circuit.

        The first `input_dim` parameters encode the input features
        via Ry rotations.  The remaining `weight_dim` parameters
        are applied as Ry gates after a CNOT‑based entanglement
        layer.  The circuit ends with a measurement of Pauli‑Y
        on the third qubit.
        """
        # Input encoding
        for i, p in enumerate(params[: self.input_dim]):
            qml.Ry(p, wires=i)

        # Entanglement layer
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])

        # Trainable rotations
        for i, p in enumerate(params[self.input_dim:]):
            qml.Ry(p, wires=i % 3)

        # Observable
        return qml.expval(qml.PauliY(2))

    def qnode(self, inputs: np.ndarray, weights: np.ndarray) -> float:
        """
        Execute the circuit for given inputs and weights.

        Parameters
        ----------
        inputs : array_like
            Input feature vector of shape (input_dim,).
        weights : array_like
            Trainable weight vector of shape (weight_dim,).

        Returns
        -------
        float
            Expectation value of the observable, used as the regression output.
        """
        fn = qml.QNode(self.circuit, self.dev, interface="autograd")
        return fn(*np.concatenate([inputs, weights]))

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper that returns a prediction for a batch of inputs
        using a fixed set of weights.

        Parameters
        ----------
        inputs : array_like
            Batch of input vectors, shape (batch_size, input_dim).

        Returns
        -------
        np.ndarray
            Array of predictions, shape (batch_size,).
        """
        # For demonstration, use zero weights. In practice these would be learned.
        zero_weights = np.zeros(self.weight_dim)
        return np.array([self.qnode(inp, zero_weights) for inp in inputs])

__all__ = ["EstimatorQNN"]
