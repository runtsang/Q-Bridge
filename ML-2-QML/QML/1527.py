"""Quantum multi‑head self‑attention built with PennyLane.

The implementation mimics the classical interface but uses a
variational circuit to realise the attention mechanism. The
parameter vectors *rotation_params* and *entangle_params* are
directly mapped to Ry, Rz, Rx and controlled‑RZ rotations.
The circuit returns the expectation values of Pauli‑Z for each
qubit, which can be interpreted as the attention scores.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class SelfAttentionEnhanced:
    """
    Variational self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be at least 2).
    dev : pennylane.Device, optional
        PennyLane device. Defaults to a ``default.qubit`` simulator.
    """

    def __init__(self, n_qubits: int, dev: qml.Device = None):
        if n_qubits < 2:
            raise ValueError("n_qubits must be >= 2 for entanglement")
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        @qml.qnode(self.dev, interface="torch")
        def circuit():
            # Apply single‑qubit rotations
            for i in range(self.n_qubits):
                qml.RY(rotation_params[3*i], wires=i)
                qml.RZ(rotation_params[3*i+1], wires=i)
                qml.RX(rotation_params[3*i+2], wires=i)

            # Entangle neighbouring qubits
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i+1])

            # Measure Pauli‑Z expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        """
        Execute the variational attention circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape ``(3 * n_qubits,)`` – parameters for Ry, Rz, Rx on each qubit.
        entangle_params : np.ndarray
            Shape ``(n_qubits - 1,)`` – parameters for CRX between adjacent qubits.
        shots : int, optional
            Number of shots for the simulator (ignored for default.qubit).

        Returns
        -------
        np.ndarray
            Expectation values of Pauli‑Z for each qubit, representing
            the attention distribution.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        # For the default.qubit device we can request a shot‑based simulation
        if isinstance(self.dev, qml.devices.SimulationDevice):
            res = circuit()
        else:
            res = circuit()
        return np.array(res)

__all__ = ["SelfAttentionEnhanced"]
