"""Quantum variational fully‑connected layer using Pennylane.

The class implements a parameterised ansatz that operates on a
multi‑qubit register.  A stack of entangling layers followed by
parameterised rotations is used to model the weight matrix of a
classical fully‑connected layer.  The circuit evaluates the
expectation value of a Pauli‑Z observable on the first qubit, which
serves as the output of the layer.  The implementation supports
automatic differentiation via Pennylane's autograd backend, enabling
gradient‑based optimisation on a simulator or real device.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Iterable


class FCL:
    """
    Variational quantum circuit mimicking a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be >= 1).
    layers : int, default 2
        Number of entangling layers in the ansatz.
    dev : pennylane.Device, optional
        Pennylane device to execute the circuit on.  If ``None``, a
        default qiskit Aer simulator is used.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        layers: int = 2,
        dev: qml.Device | None = None,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = dev or qml.device("qiskit.aer", wires=n_qubits, shots=1024)

        # Initialise parameters: one rotation per qubit per layer
        self.params = pnp.random.uniform(
            low=0, high=2 * np.pi, size=(layers, n_qubits)
        )

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params, x):
            # Encode input data into the first qubit
            qml.RY(x, wires=0)
            for layer in range(self.layers):
                for qubit in range(self.n_qubits):
                    qml.RY(params[layer, qubit], wires=qubit)
                # Entangle all neighbouring qubits
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit for a single input value.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable containing a single input value to be encoded
            into the circuit.

        Returns
        -------
        np.ndarray
            Expectation value of Pauli‑Z on the first qubit, wrapped in
            a one‑element array to match the classical interface.
        """
        if len(thetas)!= 1:
            raise ValueError("Quantum FCL expects a single input value.")
        x = thetas[0]
        expectation = self.circuit(self.params, x)
        return np.array([expectation])

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the circuit output with respect to the
        variational parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Single input value.

        Returns
        -------
        np.ndarray
            Gradient vector of shape ``(layers, n_qubits)``.
        """
        if len(thetas)!= 1:
            raise ValueError("Quantum FCL expects a single input value.")
        x = thetas[0]
        grad_fn = qml.grad(self.circuit)
        grad = grad_fn(self.params, x)
        return grad

__all__ = ["FCL"]
