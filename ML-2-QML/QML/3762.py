"""Hybrid quantum‑classical classifier – quantum module."""

from __future__ import annotations

from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# --------------------------------------------------------------------------- #
#  Quantum estimator utilities (adapted from FastBaseEstimator.py)
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluates expectation values for a parameterized circuit."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Iterable[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
#  Quantum circuit builder
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered parameter‑encoding ansatz with the same depth as its
    classical counterpart.  The function returns the circuit, the encoding
    parameters, the variational parameters, and a list of Z‑observables
    for measurement.
    """
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

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return qc, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
#  Public hybrid class (mirrors the classical side)
# --------------------------------------------------------------------------- #
class QuantumClassifierModel:
    """
    Dual‑mode classifier that can be instantiated with either a torch model
    (used by the classical FastEstimator) or a Qiskit QuantumCircuit
    (used by the quantum FastBaseEstimator).  The API for evaluation is
    identical, enabling seamless switching between regimes.
    """
    def __init__(self, model: QuantumCircuit | "nn.Module"):
        if isinstance(model, QuantumCircuit):
            self.estimator = FastBaseEstimator(model)
        else:
            # Lazy import to avoid pulling in heavy dependencies when only the
            # quantum side is needed.
            from.FastEstimator import FastEstimator as _FastEstimator
            self.estimator = _FastEstimator(model)

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Iterable[Iterable[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        return self.estimator.evaluate(
            observables, parameter_sets, shots=shots, seed=seed
        )

__all__ = ["build_classifier_circuit", "FastBaseEstimator", "QuantumClassifierModel"]
