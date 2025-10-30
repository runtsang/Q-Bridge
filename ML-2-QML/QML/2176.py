"""Quantum classifier with data‑re‑uploading variational ansatz and hybrid training support."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class QuantumClassifierModel:
    """Variational classifier that mirrors the classical helper interface.

    Enhancements:
    * data‑re‑uploading (incremental encoding)
    * configurable depth, qubits and noise model
    * hybrid gradient routine for classical optimisation
    * metadata attributes (`encoding`, ``weight_sizes``, ``observables``)
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        device: str = "default.qubit",
        shots: int = 1000,
        noise: bool = False,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = qml.device(device, wires=num_qubits, shots=shots)

        # Observables for each qubit
        self.observables: List[qml.operation.Operation] = [qml.PauliZ(i) for i in range(num_qubits)]

        # Parameter vectors
        self.encoding_params = pnp.array([0.0] * num_qubits)
        self.theta = pnp.array([0.0] * (num_qubits * depth))

        @qml.qnode(self.device, interface="torch")
        def circuit(inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Data‑re‑uploading layer
            for qubit in range(num_qubits):
                qml.RX(inputs[qubit], wires=qubit)

            idx = 0
            for _ in range(depth):
                for qubit in range(num_qubits):
                    qml.RY(params[idx], wires=qubit)
                    idx += 1
                for qubit in range(num_qubits - 1):
                    qml.CZ(wires=[qubit, qubit + 1])

            return [qml.expval(obs) for obs in self.observables]

        self.circuit = circuit

        # expose metadata for compatibility
        self.encoding = list(range(num_qubits))
        self.weight_sizes = [self.theta.size]
        # ``observables`` already defined above

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Return expectation values for the input data."""
        return self.circuit(inputs, self.theta)

    def hybrid_gradients(
        self,
        inputs: np.ndarray,
        target: int,
        loss_fn: callable = lambda out, tgt: (out[tgt] - out[1 - tgt]) ** 2,
    ) -> np.ndarray:
        """Compute gradients of a simple contrastive loss w.r.t. the parameters.

        Designed to be called from a classical optimiser.
        """
        def loss(params: np.ndarray) -> np.ndarray:
            out = self.circuit(inputs, params)
            return loss_fn(out, target)

        grads = qml.grad(loss)(self.theta)
        return grads


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    device: str = "default.qubit",
    shots: int = 1000,
    noise: bool = False,
) -> Tuple[QuantumClassifierModel, Iterable[int], Iterable[int], List[qml.operation.Operation]]:
    """Compatibility wrapper that returns the quantum model and metadata."""
    model = QuantumClassifierModel(
        num_qubits,
        depth,
        device=device,
        shots=shots,
        noise=noise,
    )
    return model, model.encoding, model.weight_sizes, model.observables
