"""Hybrid quantum neural network using Pennylane.

The :class:`EstimatorQNN` implements a variational circuit with two
entangled qubits.  Classical inputs are encoded via a feature map
(ry rotations) and the circuit parameters are optimised to minimise
a mean‑squared‑error loss against a target scalar.  The model returns
the expectation value of the Pauli‑Z observable on qubit 0.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Sequence

class EstimatorQNN:
    """Hybrid quantum neural network.

    Parameters
    ----------
    n_qubits : int, default 2
        Number of qubits in the variational circuit.
    n_layers : int, default 2
        Number of alternating rotation and entanglement layers.
    dev : qml.Device, optional
        PennyLane device to run the circuit on.  If ``None`` a
        default state‑vector simulator is used.
    """

    def __init__(self, n_qubits: int = 2, n_layers: int = 2, dev: qml.Device | None = None) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

        # Number of parameters: each layer has 3*n_qubits rotation gates
        self.n_params = 3 * n_qubits * n_layers

        # Initialise parameters randomly
        self.params = pnp.random.randn(self.n_params)

        # Build the QNode
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: Sequence[float], weights: Sequence[float]) -> float:
            # Feature map: encode inputs as Ry rotations
            for i, w in enumerate(inputs):
                qml.RY(w, wires=i)

            # Variational layers
            idx = 0
            for _ in range(self.n_layers):
                for wire in range(self.n_qubits):
                    qml.Rot(
                        weights[idx],
                        weights[idx + 1],
                        weights[idx + 2],
                        wires=wire,
                    )
                    idx += 3
                # Entanglement via CNOT on a ring
                for wire in range(self.n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])

            # Measurement: expectation of Z on qubit 0
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return the model prediction for a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            2‑D array of shape ``(batch_size, n_qubits)``.
        """
        preds = []
        for inp in inputs:
            exp = self.circuit(inp, self.params)
            preds.append(float(exp))
        return np.array(preds)

    def loss(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Mean‑squared‑error loss."""
        preds = self.predict(inputs)
        return np.mean((preds - targets) ** 2)

    def train(self, inputs: np.ndarray, targets: np.ndarray, lr: float = 0.01, epochs: int = 100):
        """Simple gradient‑descent training loop."""
        opt = qml.GradientDescentOptimizer(lr)
        for _ in range(epochs):
            self.params = opt.step(lambda w: self.loss(inputs, targets), self.params)
