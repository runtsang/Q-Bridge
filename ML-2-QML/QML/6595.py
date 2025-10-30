"""
Quantum‑enhanced estimator built with PennyLane and a Qiskit backend.
The circuit is a parameter‑shallow variational ansatz with optional
entanglement and a user‑supplied observable.  The class implements
a ``predict`` method that evaluates the expectation value for a
given classical input vector.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Sequence, Iterable, Tuple, List


class EstimatorQNN:
    """
    Variational quantum regressor.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    depth : int
        Number of variational layers.
    observable : str, optional
        Pauli string observable for measurement.  Default is a single
        ``Y`` operator on the first qubit.
    backend : str, optional
        PennyLane backend name.  Defaults to ``"qiskit"`` for a Qiskit
        simulator, but any compatible backend works.
    """

    def __init__(
        self,
        num_qubits: int = 1,
        depth: int = 2,
        observable: str = "Y",
        backend: str = "qiskit",
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.observable = observable
        self.backend = backend

        # Create a quantum device
        self.dev = qml.device(backend, wires=num_qubits)

        # Build the variational circuit as a QNode
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: Sequence[float], weights: Sequence[float]) -> float:
            # Data re-uploading: encode each input component
            for i, wire in enumerate(range(num_qubits)):
                qml.RX(inputs[i % len(inputs)], wires=wire)
            # Variational layers
            w_idx = 0
            for _ in range(depth):
                for wire in range(num_qubits):
                    qml.RY(weights[w_idx], wires=wire)
                    w_idx += 1
                # Entanglement via CZ
                for wire in range(num_qubits - 1):
                    qml.CZ(wires=[wire, wire + 1])
            # Measurement
            return qml.expval(qml.PauliZ(wires=0)) if self.observable == "Z" else qml.expval(qml.PauliY(wires=0))

        self.circuit = circuit

        # Initialise weights
        weight_len = depth * num_qubits
        self.weights = pnp.random.uniform(-np.pi, np.pi, size=weight_len)

    def predict(self, inputs: Iterable[float]) -> float:
        """
        Evaluate the circuit for a single input vector.

        Parameters
        ----------
        inputs : iterable of float
            Classical feature vector.  It will be repeated if shorter
            than ``num_qubits``.
        """
        inputs_arr = np.asarray(inputs, dtype=float)
        return float(self.circuit(inputs_arr, self.weights))

    def set_weights(self, new_weights: Sequence[float]) -> None:
        """
        Overwrite the variational parameters.

        Parameters
        ----------
        new_weights : sequence of float
            Must match the length ``depth * num_qubits``.
        """
        if len(new_weights)!= self.depth * self.num_qubits:
            raise ValueError("Incorrect weight vector length.")
        self.weights = pnp.array(new_weights, dtype=float)

    def get_weights(self) -> np.ndarray:
        """
        Return the current weight vector.
        """
        return np.array(self.weights)

__all__ = ["EstimatorQNN"]
