"""Quantum‑centric hybrid classifier using Qiskit Machine Learning.

The model exposes the same public API as the classical counterpart
(`forward` and `predict`) but internally relies on a
`qiskit_machine_learning.neural_networks.EstimatorQNN`
to evaluate a variational circuit.  The quantum circuit is built with
the same topology as the original `build_classifier_circuit`, and
its parameters are updated via the `StatevectorEstimator` primitive.
"""

from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

# --------------------------------------------------------------------------- #
# Quantum building blocks
# --------------------------------------------------------------------------- #

def _build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp.from_list([("Z" * i + "I" * (num_qubits - i - 1), 1)]) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #

class QuantumClassifierModel:
    """
    Quantum neural network classifier that wraps a Qiskit EstimatorQNN.
    The API matches the classical version, enabling side‑by‑side
    experimentation without changing the training loop.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        device: str | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.enc_params, self.weight_params, self.observables = _build_classifier_circuit(num_qubits, depth)

        # The EstimatorQNN expects a statevector estimator backend.
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.enc_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute class logits from quantum circuit.

        Parameters
        ----------
        inputs : np.ndarray
            Input data of shape (batch, num_qubits).

        Returns
        -------
        np.ndarray
            Logits of shape (batch, num_qubits).
        """
        # The EstimatorQNN returns expectation values for each observable.
        # We treat each observable as a separate class logit.
        return self.qnn.predict(inputs)

    # --------------------------------------------------------------------- #
    # Convenience API
    # --------------------------------------------------------------------- #
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply softmax to the raw logits to obtain class probabilities.
        """
        logits = self.forward(inputs)
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)
