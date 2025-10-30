"""Quantum convolutional filter implemented with a variational circuit.

The class builds a small variational circuit that encodes a
``kernel_size x kernel_size`` patch of classical data into a set of qubits.
The circuit is trained end‑to‑end using Pennylane's gradient‑based
optimizers.  The output is the average probability of measuring |1>
across all qubits.  A learnable threshold can be applied to the
probability to emulate the behaviour of the original quanvolution
filter.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


class Conv:
    """
    Quantum convolutional filter that mirrors the classical Conv module.
    The filter operates on a ``kernel_size x kernel_size`` patch and uses
    a variational circuit with parameterized rotations and a
    strongly‑entangling layer.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel.
    n_layers : int, default 2
        Number of variational layers.
    threshold : float, default 0.5
        Threshold applied to the probability of measuring |1>.
    """

    def __init__(self, kernel_size: int = 2, n_layers: int = 2, threshold: float = 0.5):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.n_layers = n_layers
        self.threshold = threshold

        # Device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Parameter shape: (n_layers, n_qubits)
        self.params_shape = (n_layers, self.n_qubits)

        # Initialize parameters
        self.params = np.random.uniform(0, 2 * np.pi, self.params_shape)

        # Build the quantum node
        @qml.qnode(self.dev, interface="autograd")
        def circuit(data, params):
            # Angle‑encoding of data
            for i, val in enumerate(data):
                qml.RX(val * np.pi, wires=i)
            # Variational layers
            for layer in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RY(params[layer, q], wires=q)
                # Entangling layer
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            # Expectation value of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data: np.ndarray) -> float:
        """
        Execute the variational circuit on a single data patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flat = data.reshape(-1).astype(float)
        expvals = self.circuit(flat, self.params)
        # Convert expectation values to probabilities of |1>
        probs = 0.5 * (1 - np.array(expvals))
        return probs.mean()

    def loss(self, data_batch: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute the mean‑squared‑error loss over a batch.

        Parameters
        ----------
        data_batch : np.ndarray
            Batch of data patches with shape (batch, kernel_size, kernel_size).
        targets : np.ndarray
            Target probabilities.

        Returns
        -------
        float
            Loss value.
        """
        preds = np.array([self.run(d) for d in data_batch])
        return ((preds - targets) ** 2).mean()

    def train_step(
        self,
        data_batch: np.ndarray,
        targets: np.ndarray,
        lr: float = 0.01,
    ) -> float:
        """
        Perform one gradient‑descent step using Pennylane's autograd.

        Parameters
        ----------
        data_batch : np.ndarray
            Batch of input data.
        targets : np.ndarray
            Target probabilities.
        lr : float, default 0.01
            Learning rate.

        Returns
        -------
        float
            Loss value after the step.
        """
        # Gradient of loss w.r.t. parameters
        grad_fn = qml.grad(self.loss)
        grads = grad_fn(data_batch, targets)
        # Update parameters
        self.params = self.params - lr * grads
        return self.loss(data_batch, targets)

    def evaluate(self, data_batch: np.ndarray) -> np.ndarray:
        """
        Evaluate the filter on a batch of data patches.

        Parameters
        ----------
        data_batch : np.ndarray
            Batch of data patches.

        Returns
        -------
        np.ndarray
            Array of predicted probabilities.
        """
        return np.array([self.run(d) for d in data_batch])


__all__ = ["Conv"]
