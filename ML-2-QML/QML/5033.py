"""Quantum implementation of the hybrid model.

The class mirrors the classical ``HybridModel`` interface while performing
the entire computation on a quantum circuit.  It uses the same
``build_classifier_circuit`` helper from reference pair 3 and the
``FastBaseEstimator`` from reference pair 4 for efficient state‑vector
evaluation.  The resulting expectation values can be directly fed into
a downstream classical classifier or used as the logits of a quantum
classifier.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

# --------------------------------------------------------------------------- #
# Helper that builds a classifier circuit (reference pair 3)
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_qubits: int,
                             depth: int) -> Tuple[QuantumCircuit,
                                                  Iterable[ParameterVector],
                                                  Iterable[ParameterVector],
                                                  List[SparsePauliOp]]:
    """
    Construct a simple layered ansatz with explicit encoding and variational
    parameters.  The returned objects are:

    * circuit – the parameterised quantum circuit
    * encoding – parameters used for data encoding
    * weights   – variational parameters
    * observables – Pauli‑Z measurements for each qubit
    """
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

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
# Fast estimator from reference pair 4
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Iterable[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[SparsePauliOp],
                 parameter_sets: Iterable[Iterable[float]]
                 ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


# --------------------------------------------------------------------------- #
# Quantum hybrid model
# --------------------------------------------------------------------------- #

class HybridModel:
    """
    Quantum counterpart of the classical ``HybridModel``.

    Parameters
    ----------
    num_features : int
        Number of input features (qubits).
    depth : int
        Variational depth of the ansatz.
    """
    def __init__(self, num_features: int, depth: int = 2) -> None:
        circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_features, depth
        )
        self.estimator = FastBaseEstimator(circuit)

    def encode(self, inputs: Iterable[float]) -> List[float]:
        """Return the parameter values for the encoding registers."""
        return list(inputs)

    def run(self,
            parameter_sets: Iterable[Iterable[float]]
            ) -> List[List[complex]]:
        """
        Evaluate the quantum circuit for each parameter set.

        The first ``len(self.encoding)`` parameters are treated as data
        encoding, the remaining as variational weights.  The method returns
        the expectation values of the observables.
        """
        return self.estimator.evaluate(self.observables, parameter_sets)

    def predict(self,
                inputs: Iterable[Iterable[float]]
                ) -> List[List[complex]]:
        """
        Convenience wrapper that accepts raw feature vectors and returns the
        classification logits.  Each input vector is padded with a dummy
        weight slice (all zeros) so that the parameter count matches the
        estimator's expectations.
        """
        full_params = [
            list(inp) + [0.0] * len(self.weights)
            for inp in inputs
        ]
        return self.run(full_params)


__all__ = ["HybridModel",
           "build_classifier_circuit",
           "FastBaseEstimator"]
