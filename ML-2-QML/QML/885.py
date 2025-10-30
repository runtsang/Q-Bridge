"""Quantum classifier with a flexible ansatz and hybrid training support.

Features added compared to the seed:
- Data‑encoding via amplitude encoding or angle encoding.
- Entangling layers using CZ and CX gates.
- Parameter‑shift gradient estimation.
- Integration with Pennylane's autograd for efficient back‑prop.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class QuantumClassifierModel:
    """
    Quantum circuit factory that returns a Pennylane QNode ready for training.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        encoding: str = "angle",
        ansatz: str = "cz_cy",
        device_name: str = "default.qubit",
        shots: int = 1024,
    ):
        """
        Parameters
        ----------
        num_qubits: int
            Number of qubits / input features.
        depth: int
            Number of variational layers.
        encoding: str
            Type of data encoding: ``angle`` or ``amplitude``.
        ansatz: str
            Entanglement pattern: ``cz_cy`` or ``cx``.
        device_name: str
            Pennylane device to use.
        shots: int
            Number of shots for expectation estimation.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = encoding
        self.ansatz = ansatz
        self.device = qml.device(device_name, wires=num_qubits, shots=shots)
        self.params = pnp.random.uniform(0, 2 * np.pi, size=(depth, num_qubits), requires_grad=True)

    def _encode(self, x: np.ndarray) -> None:
        """Apply the chosen data‑encoding scheme."""
        if self.encoding == "angle":
            for i, val in enumerate(x):
                qml.RX(val, wires=i)
        elif self.encoding == "amplitude":
            # amplitude encoding: prepare state |x⟩
            qml.QubitStateVector(x, wires=range(self.num_qubits))
        else:
            raise ValueError(f"Unsupported encoding {self.encoding}")

    def _ansatz_layer(self, layer_idx: int) -> None:
        """Apply a single variational layer."""
        for qubit in range(self.num_qubits):
            qml.RY(self.params[layer_idx, qubit], wires=qubit)
        if self.ansatz == "cz_cy":
            for qubit in range(self.num_qubits - 1):
                qml.CZ(qubit, qubit + 1)
                qml.CY(qubit, qubit + 1)
        elif self.ansatz == "cx":
            for qubit in range(self.num_qubits - 1):
                qml.CNOT(qubit, qubit + 1)

    def circuit(self, x: np.ndarray) -> np.ndarray:
        """Return the expectation values of Z on each qubit."""
        @qml.qnode(self.device, interface="autograd")
        def qnode(data: np.ndarray):
            self._encode(data)
            for layer in range(self.depth):
                self._ansatz_layer(layer)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        return qnode(x)

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        encoding: str = "angle",
        ansatz: str = "cz_cy",
    ) -> Tuple[qml.QNode, Iterable[float], Iterable[float], List[qml.operation.Tensor]]:
        """
        Factory mirroring the quantum helper signature.

        Returns
        -------
        qnode, encoding, weight_sizes, observables
        """
        model = QuantumClassifierModel(num_qubits, depth, encoding, ansatz)
        # For compatibility, expose parameters as a flat list of floats
        encoding = list(range(num_qubits))
        weight_sizes = [p.size for p in model.params]
        observables = [qml.PauliZ(i) for i in range(num_qubits)]
        return model.circuit, encoding, weight_sizes, observables

    def train(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        epochs: int = 50,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train the variational circuit using a simple gradient‑descent loop.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        loss_history: List[float] = []

        for epoch in range(epochs):
            def loss_fn():
                preds = self.circuit(data)
                # Convert logits to probabilities via softmax
                probs = pnp.exp(preds) / pnp.sum(pnp.exp(preds), axis=0)
                # Cross‑entropy with one‑hot labels
                ce = -pnp.sum(labels * pnp.log(probs + 1e-8), axis=0)
                return pnp.mean(ce)

            loss = loss_fn()
            loss_history.append(loss.item())
            if verbose:
                print(f"[{epoch+1:02d}] loss={loss:.4f}")

            opt.step(self.params)

        return loss_history

__all__ = ["QuantumClassifierModel"]
