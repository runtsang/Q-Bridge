"""Hybrid quantum estimator that evaluates parametric circuits with observables
and supports a quantum classifier ansatz."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.providers import Backend
from qiskit import Aer

class FastHybridQuantumEstimator:
    """Evaluator that runs a parametric QuantumCircuit and returns expectation values.

    It mirrors the classical FastHybridEstimator but operates on quantum states.
    """
    def __init__(self,
                 circuit: QuantumCircuit,
                 backend: Backend | None = None,
                 shots: int | None = None) -> None:
        self.circuit = circuit
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots or 1024
        self.params = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with parameters bound to the supplied values."""
        if len(parameter_values)!= len(self.params):
            raise ValueError("Parameter count mismatch between circuit and values.")
        mapping = dict(zip(self.params, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[SparsePauliOp],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Return expectation values for each parameter set and observable.

        Uses the Statevector simulator for exact results.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self._bind(params)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_noise(self,
                            observables: Iterable[SparsePauliOp],
                            parameter_sets: Sequence[Sequence[float]],
                            *,
                            shots: int,
                            seed: int | None = None) -> List[List[complex]]:
        """Same as evaluate but adds shot‑noise to each expectation value.

        The noise model is Gaussian with variance 1/shots, mimicking measurement
        statistics of a real backend.
        """
        exact = self.evaluate(observables, parameter_sets)
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in exact:
            noisy_row = [float(rng.normal(val.real, max(1e-6, 1 / shots))) for val in row]
            noisy.append(noisy_row)
        return noisy

def build_quantum_classifier_circuit(num_qubits: int,
                                     depth: int) -> Tuple[QuantumCircuit,
                                                          Iterable[ParameterVector],
                                                          Iterable[ParameterVector],
                                                          List[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters.

    Mirrors the helper in reference 4 but returns the parameters separately for
    convenience during training or evaluation.
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

__all__ = ["FastHybridQuantumEstimator",
           "build_quantum_classifier_circuit"]
