"""Quantum kernel‑based classifier using Qiskit VQC.

The implementation mirrors the classical interface of
``KernelClassifier`` but replaces the kernel evaluation with a
parameterised variational circuit.  The circuit is built using the
``build_classifier_circuit`` helper from reference pair 2, which
provides an encoding layer and a depth‑controlled variational block.
"""

from __future__ import annotations

from typing import List

import numpy as np
from qiskit import Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.algorithms import VQC
from qiskit.algorithms.optimizers import COBYLA
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------
# Circuit factory (same as reference pair 2)
# ----------------------------------------------------------------------
def build_classifier_circuit(
    num_qubits: int, depth: int
) -> tuple["QuantumCircuit", List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Return a Qiskit circuit, its encoding and weight parameters,
    and the observables used for measurement."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp

    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


# ----------------------------------------------------------------------
# Unified quantum classifier
# ----------------------------------------------------------------------
class KernelClassifier:
    """Quantum kernel‑based classifier.

    Parameters
    ----------
    num_qubits : int, default=4
        Number of qubits in the circuit.
    depth : int, default=2
        Depth of the variational ansatz.
    backend : qiskit.providers.Backend, optional
        Backend used for state‑vector or qasm simulation.  If ``None``,
        the local qasm simulator is used.
    """
    def __init__(
        self,
        num_qubits: int = 4,
        depth: int = 2,
        backend=None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Build the circuit and extract parameters
        (
            self.circuit,
            self.encoding,
            self.weights,
            self.observables,
        ) = build_classifier_circuit(num_qubits, depth)

        # VQC instance
        self.vqc = VQC(
            quantum_instance=self.backend,
            feature_map=self.circuit,
            ansatz=self.circuit,
            optimizer=COBYLA(maxiter=200),
            training_dataset=None,
        )

        self.scaler = StandardScaler()
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the variational circuit."""
        X = self.scaler.fit_transform(X)
        self.X_train = X
        self.y_train = y
        training_dataset = [(x, label) for x, label in zip(X, y)]
        self.vqc.fit(training_dataset)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data."""
        X = self.scaler.transform(X)
        return self.vqc.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the raw decision scores."""
        X = self.scaler.transform(X)
        return self.vqc.decision_function(X)

__all__ = ["KernelClassifier"]
