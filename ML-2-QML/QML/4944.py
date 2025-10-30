"""Quantum classifier circuit mirroring the classical helper.

The circuit uses a data‑encoding layer, a layered variational ansatz,
and a sampler sub‑circuit inspired by the Qiskit SamplerQNN example.
FastBaseEstimator evaluates expectation values of Pauli‑Z observables
and can add Gaussian shot noise to emulate measurement uncertainty.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector, BaseOperator
import numpy as np


# --------------------------------------------------------------------------- #
#  Sampler sub‑circuit
# --------------------------------------------------------------------------- #

def _sampler_circuit() -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Return a 2‑qubit sampler circuit from the Qiskit example."""
    inputs = ParameterVector("x", 2)
    weights = ParameterVector("w", 4)

    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    return qc, inputs, weights


# --------------------------------------------------------------------------- #
#  Main classifier circuit
# --------------------------------------------------------------------------- #

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered ansatz with data encoding and an embedded sampler.

    Returns:
        circuit: QuantumCircuit
        encoding: list of ParameterVector objects for data encoding
        weights: list of ParameterVector objects for variational parameters
        observables: list of SparsePauliOp measuring Z on each qubit
    """
    # Data encoding
    encoding = ParameterVector("x", num_qubits)

    # Variational weights for the main ansatz
    weights = ParameterVector("theta", num_qubits * depth)

    # Main circuit
    qc = QuantumCircuit(num_qubits)

    # Encode data with RX rotations
    for qubit in range(num_qubits):
        qc.rx(encoding[qubit], qubit)

    # Layered ansatz
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Attach sampler sub‑circuit on two ancilla qubits
    sampler, _, _ = _sampler_circuit()
    qc.append(sampler, [num_qubits, num_qubits + 1])

    # Observables: Z on each data qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return qc, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
#  Fast estimator for expectation values
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

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Iterable[Iterable[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                rng.normal(float(val.real), max(1e-6, 1 / shots))
                + 1j * rng.normal(float(val.imag), max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["build_classifier_circuit", "FastBaseEstimator", "FastEstimator"]
