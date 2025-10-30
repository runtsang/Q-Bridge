"""Quantum implementation of a fully connected layer using a variational circuit.

The circuit consists of a stack of parameterised Ry rotations followed by
CNOT entangling gates.  The expectation value of a Pauli‑Z measurement
on the first qubit is returned.  The ``run`` method accepts a flattened
list of parameters matching the circuit depth and number of qubits.
"""

import pennylane as qml
import numpy as np
from typing import Iterable, Sequence


class FCL:
    """
    Variational quantum circuit emulating a fully connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    layers : int
        Number of variational layers.
    dev : pennylane.Device, optional
        Quantum device to execute the circuit on.  Defaults to a
        ``default.qubit`` simulator.
    """

    def __init__(self, n_qubits: int, layers: int, dev: qml.Device = None) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)
        self.n_params = layers * n_qubits

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: np.ndarray) -> np.ndarray:
            for l in range(layers):
                for q in range(n_qubits):
                    qml.RY(params[l, q], wires=q)
                # Entangle all neighbouring qubits
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                # Wrap‑around entanglement
                qml.CNOT(wires=[n_qubits - 1, 0])
            # Measure expectation of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the variational circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flattened list of parameters of length ``layers * n_qubits``.
        """
        thetas = np.array(thetas, dtype=np.float32).reshape(self.layers, self.n_qubits)
        expectation = self.circuit(thetas)
        return np.array([expectation])

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the expectation value w.r.t. the parameters
        using the parameter‑shift rule.

        Parameters
        ----------
        thetas : Iterable[float]
            Flattened list of parameters.
        """
        thetas = np.array(thetas, dtype=np.float32).reshape(self.layers, self.n_qubits)
        grad = qml.grad(self.circuit)(thetas)
        return grad.reshape(-1)
