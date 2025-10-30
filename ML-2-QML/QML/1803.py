"""FullyConnectedLayerGen402 – a quantum variational layer built with Pennylane.

The class exposes a QNode that maps a flat list of rotation angles
to an expectation value of a single observable on a multi‑qubit
entangled circuit.  The circuit is fully differentiable via
Pennylane's autograd engine, allowing seamless integration in hybrid
learning loops.
"""

from __future__ import annotations

from typing import Iterable, List

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class FullyConnectedLayerGen402:
    """
    Parameters
    ----------
    n_qubits : int
        Number of qubits in the variational circuit.
    entanglement : str, optional
        Entanglement pattern – 'linear', 'circular', or 'full'.
        Defaults to 'linear'.
    observable : str, optional
        Observable used for expectation.  Currently supports
        'PauliZ' on the first qubit.
    shots : int, optional
        Number of shots for simulation.  Defaults to 1000.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        entanglement: str = "linear",
        observable: str = "PauliZ",
        shots: int = 1000,
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.device = qml.device("default.qubit", wires=n_qubits)

        # Build the QNode
        @qml.qnode(self.device, interface="autograd", diff_method="backprop")
        def circuit(params):
            # Parameter‑shifted rotation layers
            for i, wire in enumerate(range(n_qubits)):
                qml.RY(params[i], wires=wire)

            # Entanglement
            if entanglement == "linear":
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            elif entanglement == "circular":
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            elif entanglement == "full":
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])

            # Measurement observable
            if observable == "PauliZ":
                return qml.expval(qml.PauliZ(wires=0))
            else:
                raise ValueError("Unsupported observable")

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the variational circuit for a flat list of parameters.

        Returns
        -------
        expectation : np.ndarray
            A single‑element array containing the expectation value.
        """
        params = np.array(list(thetas), dtype=np.float64)
        if params.size!= self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} parameters, got {params.size}"
            )
        exp_val = self.circuit(params)
        return np.array([exp_val])

__all__ = ["FullyConnectedLayerGen402"]
