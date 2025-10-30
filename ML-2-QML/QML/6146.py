"""Quantum QCNN with depth‑aware entangling and measurement‑based pooling."""

import pennylane as qml
from pennylane import numpy as pnp

class QCNNModelExt:
    """Quantum counterpart of :class:`QCNNModelExt`.

    Builds a parameter‑efficient QCNN circuit with a configurable depth.
    """

    def __init__(self, depth: int = 3, qubits: int = 8, seed: int = 123) -> None:
        self.depth = depth
        self.qubits = qubits
        self.seed = seed
        qml.set_options(device="default.qubit", wires=qubits)
        self.dev = qml.device("default.qubit", wires=qubits)
        self.circuit, self.weight_shape = self._build_circuit()

    # ------------------------------------------------------------------
    # Feature map: simple RX rotations
    # ------------------------------------------------------------------
    def _feature_map(self, x):
        for i, val in enumerate(x):
            qml.RX(val, wires=i)

    # ------------------------------------------------------------------
    # Parameter‑efficient entangling block
    # ------------------------------------------------------------------
    def _entangle_block(self, params, wires):
        for i in range(0, len(wires) - 1, 2):
            qml.CNOT(wires[i], wires[i + 1])
            qml.RZ(params[i], wires[i])
            qml.RY(params[i + 1], wires[i + 1])
            qml.CNOT(wires[i], wires[i + 1])

    # ------------------------------------------------------------------
    # Measurement‑based pooling
    # ------------------------------------------------------------------
    def _pooling_block(self, wires, pool_params):
        for i in range(0, len(wires) - 1, 2):
            qml.CNOT(wires[i], wires[i + 1])
            qml.RZ(pool_params[i], wires[i])
            qml.RY(pool_params[i + 1], wires[i + 1])
            qml.measure(wires[i + 1])  # collapse and reset

    # ------------------------------------------------------------------
    # Build the full circuit
    # ------------------------------------------------------------------
    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, weights):
            self._feature_map(x)
            w_iter = iter(weights)
            for _ in range(self.depth):
                # Convolution layer
                self._entangle_block(next(w_iter), list(range(self.qubits)))
                # Pooling layer
                self._pooling_block(list(range(self.qubits)), next(w_iter))
            # Observable: Z on first qubit
            return qml.expval(qml.PauliZ(0))

        # Weight shape: depth × (qubits + qubits) parameters
        weight_shape = (self.depth, self.qubits * 2)
        return circuit, weight_shape

    def forward(self, x, weights):
        """Execute the QCNN circuit.

        Parameters
        ----------
        x : array_like
            Input feature vector of length ``qubits``.
        weights : array_like
            Weight matrix of shape ``weight_shape``.

        Returns
        -------
        float
            Expectation value of Pauli‑Z on qubit 0.
        """
        return self.circuit(x, weights)
