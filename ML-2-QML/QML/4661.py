from __future__ import annotations

from typing import Iterable, Sequence, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

# --------------------------------------------------------------------------- #
# Quantum classifier ansatz (variational circuit)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered variational ansatz with explicit feature encoding.
    The circuit is compatible with the classical counterpart defined in
    :mod:`ml_code` via the ``FraudDetectionHybrid`` class.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    # Feature encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Observables: local Z measurements
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
    ]
    return qc, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
# Fast estimator for quantum circuits
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """
    Evaluates a parametrized quantum circuit by compiling it into a statevector
    and computing expectation values of the supplied observables.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


# --------------------------------------------------------------------------- #
# Hybrid class that exposes the quantum model and estimator
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid:
    """
    Quantum implementation of the fraud‑detection model.
    It builds a variational circuit and offers an estimator that returns
    expectation values for the local Z observables.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.estimator = FastBaseEstimator(self.circuit)

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[SparsePauliOp] | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set.
        If ``observables`` is None, the default local Z observables are used.
        """
        if observables is None:
            observables = self.observables
        return self.estimator.evaluate(observables, parameter_sets)


__all__ = ["build_classifier_circuit", "FastBaseEstimator", "FraudDetectionHybrid"]
