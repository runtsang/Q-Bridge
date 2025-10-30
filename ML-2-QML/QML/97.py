"""Quantum convolutional filter using a variational circuit.

The module implements a class ``ConvFilter`` that mimics the behaviour
of the original quanvolution filter but with a learnable variational
circuit.  The circuit consists of a layer of parameterized RX gates,
followed by a fixed entangling pattern and a final measurement of
Pauli‑Z on each qubit.  The probability of measuring |1> is
computed as (1 - expectation)/2.

The public API matches the original: ``Conv()`` returns an instance
with a ``run(data)`` method that accepts a 2‑D array and outputs a
float.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class ConvFilter:
    """
    Variational quanvolution filter.

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel.
    shots : int, default 1000
        Number of shots for statevector simulation.
    threshold : float, default 0.0
        Threshold used to discretise input data into gate angles.
    """

    def __init__(self, kernel_size: int, shots: int = 1000, threshold: float = 0.0):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)

        # initialise trainable parameters (not used for inference but kept for completeness)
        self.params = pnp.random.randn(self.n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(param_vector):
            # parameterised RX gates
            for i in range(self.n_qubits):
                qml.RX(param_vector[i], wires=i)
            # fixed entangling pattern: nearest‑neighbour CNOTs in a line
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # measure expectation of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def _data_to_params(self, data: np.ndarray) -> pnp.ndarray:
        """
        Convert input data into a parameter vector for the circuit.
        Values above ``threshold`` are mapped to ``π``; otherwise to ``0``.
        """
        flat = data.reshape(-1)
        param_vec = np.where(flat > self.threshold, np.pi, 0.0)
        return param_vec

    def run(self, data: np.ndarray) -> float:
        """
        Run the variational circuit on classical data.

        Parameters
        ----------
        data : array‑like, shape (kernel_size, kernel_size)
            Input patch.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        param_vec = self._data_to_params(data)
        expvals = self.circuit(param_vec)
        probs = [(1 - e) / 2 for e in expvals]
        return np.mean(probs)

def Conv(kernel_size: int = 2, shots: int = 1000, threshold: float = 0.0):
    """
    Factory that returns a ``ConvFilter`` instance.
    """
    return ConvFilter(kernel_size, shots=shots, threshold=threshold)

__all__ = ["ConvFilter", "Conv"]
