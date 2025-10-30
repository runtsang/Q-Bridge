"""EstimatorQNNModel – quantum variational regressor using Pennylane.

The quantum model implements a parameterised circuit that maps a 4‑qubit input
to a single observable expectation value.  It uses entangling CNOT layers
and a parameter‑shift rule for gradient evaluation, making it compatible
with both local simulators and real quantum backends.

Usage
-----
>>> from EstimatorQNN__gen252 import EstimatorQNN
>>> qmodel = EstimatorQNN()
>>> inputs = np.array([0.1, 0.2, 0.3, 0.4])
>>> weights = np.random.randn(qmodel.num_qubits * 2)
>>> pred = qmodel(inputs, weights)
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Any

__all__ = ["EstimatorQNN"]


class EstimatorQNN:
    """Variational quantum circuit for regression.

    Parameters
    ----------
    num_qubits : int, default 4
        Number of qubits in the circuit.
    device : pennylane.Device, optional
        Pennylane device; defaults to the local simulator.
    """

    def __init__(self, num_qubits: int = 4, device: Any = None) -> None:
        self.num_qubits = num_qubits
        self.device = device or qml.device("default.qubit", wires=num_qubits)
        self.qnode = qml.QNode(self._circuit, self.device, interface="numpy", diff_method="parameter-shift")

    def _circuit(self, inputs: np.ndarray, weights: np.ndarray) -> float:
        """Parameterized circuit returning the expectation of Pauli‑Z on qubit 0."""
        # Encode inputs as rotations
        for i in range(self.num_qubits):
            qml.RY(inputs[i], wires=i)
        # Apply weight rotations
        for i in range(self.num_qubits):
            qml.RZ(weights[i], wires=i)
        # Entangling layer
        for i in range(self.num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    def __call__(self, inputs: np.ndarray, weights: np.ndarray) -> float:
        """Evaluate the circuit."""
        return self.qnode(inputs, weights)

    def gradient(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return the gradient of the expectation value w.r.t. the weights."""
        return self.qnode.gradient(inputs, weights)
