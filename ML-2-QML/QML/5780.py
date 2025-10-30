"""Hybrid quantum sampler‑classifier helper combining the seed SamplerQNN and QuantumClassifierModel circuits.

The class builds a composite quantum circuit that first implements the
parameterised sampler (as in SamplerQNN.py) and then appends a layered
ansatz for classification (as in QuantumClassifierModel.py).  Two
independent quantum‑machine‑learning objects – a SamplerQNN and a
ClassifierQNN – are instantiated and exposed through the same API.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN, ClassifierQNN
from qiskit.primitives import Sampler, Estimator


class SamplerQNNGen169:
    """Combined sampler‑classifier quantum helper.

    The constructor builds two separate variational circuits:
    * A sampler circuit that produces a probability distribution over two
      computational basis states.
    * A classifier circuit that maps input states to class logits via
      expectation values of Pauli‑Z operators.

    Both circuits share the same input parameters but have distinct weight
    parameter vectors, allowing independent optimisation of sampling and
    classification.
    """

    def __init__(self, num_qubits: int = 2, depth: int = 2, backend: Any | None = None) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("statevector_simulator")

        # --- Sampler part ----------------------------------------------------
        self.input_params = ParameterVector("x", num_qubits)
        self.sampler_weights = ParameterVector("theta_s", 4)

        qc_sampler = QuantumCircuit(num_qubits)
        qc_sampler.ry(self.input_params[0], 0)
        qc_sampler.ry(self.input_params[1], 1)
        qc_sampler.cx(0, 1)
        qc_sampler.ry(self.sampler_weights[0], 0)
        qc_sampler.ry(self.sampler_weights[1], 1)
        qc_sampler.cx(0, 1)
        qc_sampler.ry(self.sampler_weights[2], 0)
        qc_sampler.ry(self.sampler_weights[3], 1)

        sampler_primitive = Sampler(backend=self.backend)
        self.sampler_qnn = SamplerQNN(
            circuit=qc_sampler,
            input_params=self.input_params,
            weight_params=self.sampler_weights,
            sampler=sampler_primitive,
        )

        # --- Classifier part -------------------------------------------------
        self.classifier_input = ParameterVector("x_c", num_qubits)
        self.classifier_weights = ParameterVector("theta_c", num_qubits * depth)

        qc_classifier = QuantumCircuit(num_qubits)
        for param, qubit in zip(self.classifier_input, range(num_qubits)):
            qc_classifier.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc_classifier.ry(self.classifier_weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc_classifier.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        estimator_primitive = Estimator(backend=self.backend)
        self.classifier_qnn = ClassifierQNN(
            circuit=qc_classifier,
            input_params=self.classifier_input,
            weight_params=self.classifier_weights,
            estimator=estimator_primitive,
        )

    def sample(self, inputs: np.ndarray, sampler_weights: np.ndarray) -> np.ndarray:
        """Return the sampled probability distribution from the quantum sampler."""
        return self.sampler_qnn.sample(inputs=inputs, weights=sampler_weights)

    def classify(self, inputs: np.ndarray, classifier_weights: np.ndarray) -> np.ndarray:
        """Return class logits (expectation values) from the quantum classifier."""
        return self.classifier_qnn.predict(inputs=inputs, weights=classifier_weights)

    def __repr__(self) -> str:
        return (
            f"SamplerQNNGen169(num_qubits={self.num_qubits}, depth={self.depth}, "
            f"backend={self.backend.__class__.__name__})"
        )


__all__ = ["SamplerQNNGen169"]
