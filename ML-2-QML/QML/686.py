"""EstimatorQNN – advanced variational quantum regressor.

This module implements a 3‑qubit variational quantum circuit using
Pennylane.  Inputs are encoded via RX rotations, followed by two
entangling layers of parameterised rotations and CNOT gates.  The
circuit returns the expectation value of Pauli‑Z on each qubit,
providing a three‑dimensional output vector that can be used for
regression or as part of a hybrid training loop.

Typical usage:

>>> from EstimatorQNN__gen350 import EstimatorQNN
>>> model = EstimatorQNN()
>>> import numpy as np
>>> x = np.array([0.5, -0.2, 0.1])  # 3‑dimensional input
>>> y = model(x)  # y is a 3‑element array of expectation values
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Sequence


class EstimatorQNN:
    """
    Variational quantum neural network.

    Parameters
    ----------
    input_dim : int, default 3
        Number of input features (must be <= num_qubits).
    num_qubits : int, default 3
        Number of qubits in the circuit.
    layers : int, default 2
        Number of variational layers.
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_qubits: int = 3,
        layers: int = 2,
    ) -> None:
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.layers = layers

        # Device and circuit definition
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Initialise trainable parameters
        # Shape: (layers, num_qubits, 2) for (RY, RZ) angles
        self.weight_params = pnp.zeros((layers, num_qubits, 2), requires_grad=True)

        # QNode wrapping the circuit
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, input_params: Sequence[float], weight_params: np.ndarray) -> Sequence[float]:
        """
        Variational circuit.

        Parameters
        ----------
        input_params : array_like
            Input encoding values.
        weight_params : ndarray
            Trainable rotation angles.

        Returns
        -------
        list[float]
            Expectation values of Pauli‑Z on each qubit.
        """
        # Input encoding – RX rotations on the first `input_dim` qubits
        for i, val in enumerate(input_params):
            qml.RX(val, wires=i % self.num_qubits)

        # Variational layers
        for layer in range(self.layers):
            for q in range(self.num_qubits):
                qml.RY(weight_params[layer, q, 0], wires=q)
                qml.RZ(weight_params[layer, q, 1], wires=q)
            # Entanglement – linear chain with wrap‑around
            for q in range(self.num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[self.num_qubits - 1, 0])

        # Measurement – expectation of Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def __call__(self, inputs: Sequence[float]) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        inputs : array_like
            Input vector of length `input_dim`.

        Returns
        -------
        ndarray
            Output vector of length `num_qubits` (Pauli‑Z expectations).
        """
        if len(inputs) > self.input_dim:
            raise ValueError(f"Expected at most {self.input_dim} input values, got {len(inputs)}.")
        return self.qnode(np.array(inputs), self.weight_params)

__all__ = ["EstimatorQNN"]
