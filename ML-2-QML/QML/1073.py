"""Quantum QCNN using Pennylane with convolution and pooling layers.

The circuit implements a 2‑qubit convolution unit followed by a 2‑qubit pooling
unit, stacked according to the original design.  The class exposes a
`forward` method that evaluates the expectation value of a Z observable on
the last qubit, enabling integration into classical training loops.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import List

class QCNNHybrid:
    """
    Quantum QCNN implemented with Pennylane.
    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit (default 8).
    device : str or pennylane.Device
        Backend device; defaults to 'default.qubit'.
    """

    def __init__(self, n_qubits: int = 8, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)
        self.params = self._init_params()
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _init_params(self) -> dict:
        """Initialize learnable parameters for conv and pool layers."""
        params = {}
        # Convolution layers: 3 params per 2‑qubit block
        conv_layers = 3  # 8 -> 4 -> 2 qubits
        for i in range(conv_layers):
            params[f"c{i+1}"] = pnp.random.uniform(0, 2 * np.pi, 3)
        # Pooling layers: 3 params per 2‑qubit block
        pool_layers = 3
        for i in range(pool_layers):
            params[f"p{i+1}"] = pnp.random.uniform(0, 2 * np.pi, 3)
        return params

    def _conv_block(self, params: np.ndarray, wires: List[int]):
        """Two‑qubit convolution unit."""
        qml.RZ(-np.pi / 2, wires=wires[1])
        qml.CNOT(*wires)
        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(*wires[::-1])
        qml.RY(params[2], wires=wires[1])
        qml.CNOT(*wires)
        qml.RZ(np.pi / 2, wires=wires[0])

    def _pool_block(self, params: np.ndarray, wires: List[int]):
        """Two‑qubit pooling unit."""
        qml.RZ(-np.pi / 2, wires=wires[1])
        qml.CNOT(*wires)
        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(*wires[::-1])
        qml.RY(params[2], wires=wires[1])

    def _circuit(self, x: np.ndarray):
        """Full quantum circuit with feature map and ansatz."""
        # Feature map: simple Z rotation
        for i, val in enumerate(x):
            qml.RZ(val, wires=i)
        # First conv layer
        for i in range(0, self.n_qubits, 2):
            self._conv_block(self.params["c1"], wires=[i, i + 1])
        # First pool layer
        for i in range(0, self.n_qubits, 4):
            self._pool_block(self.params["p1"], wires=[i, i + 2])
        # Second conv layer (4 qubits)
        for i in range(0, self.n_qubits, 4):
            self._conv_block(self.params["c2"], wires=[i, i + 2])
        # Second pool layer (2 qubits)
        for i in range(0, self.n_qubits, 8):
            self._pool_block(self.params["p2"], wires=[i, i + 4])
        # Third conv layer (2 qubits)
        self._conv_block(self.params["c3"], wires=[self.n_qubits - 2, self.n_qubits - 1])
        # Third pool layer
        self._pool_block(self.params["p3"], wires=[self.n_qubits - 2, self.n_qubits - 1])
        # Measurement
        return qml.expval(qml.PauliZ(self.n_qubits - 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the circuit and return a probability."""
        val = self.qnode(x)
        return (val + 1) / 2  # map expectation to [0,1]

def QCNNHybridFactory(**kwargs) -> QCNNHybrid:
    """Convenience factory for creating a QCNNHybrid instance."""
    return QCNNHybrid(**kwargs)

__all__ = ["QCNNHybrid", "QCNNHybridFactory"]
