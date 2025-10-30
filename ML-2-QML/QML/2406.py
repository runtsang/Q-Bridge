"""
Quantum sampler that mirrors the original SamplerQNN but with a deeper,
entangling circuit and integrated shot‑noise support.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """
    Evaluate expectation values of observables for a parametrized circuit.
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
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """
    Adds optional Gaussian shot noise to the deterministic expectation values.
    """
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
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
                complex(
                    rng.normal(float(val.real), max(1e-6, 1 / shots)),
                    rng.normal(float(val.imag), max(1e-6, 1 / shots)),
                )
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# UnifiedSamplerQNN – quantum sampler
# --------------------------------------------------------------------------- #
class UnifiedSamplerQNN:
    """
    Parameterised quantum circuit with depth‑controlled entanglement and
    a wrapper that exposes a SamplerQNN interface for state‑vector sampling.
    """
    def __init__(self, num_qubits: int = 2, depth: int = 2) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.inputs = ParameterVector("input", num_qubits)
        self.weights = ParameterVector("weight", num_qubits * depth)
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qsampler = QSamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Input rotations
        for i in range(self.num_qubits):
            qc.ry(self.inputs[i], i)
        # Entangling layers with wrap‑around connectivity
        for d in range(self.depth):
            for i in range(self.num_qubits):
                qc.ry(self.weights[d * self.num_qubits + i], i)
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(self.num_qubits - 1, 0)  # wrap‑around entanglement
        return qc

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable,
        optionally adding shot‑noise to the results.
        """
        estimator = FastEstimator(self.circuit)
        return estimator.evaluate(
            observables, parameter_sets, shots=shots, seed=seed
        )

    def sample(self, parameter_set: Sequence[float], num_shots: int = 1024) -> List[int]:
        """
        Return measurement outcomes sampled from the circuit for a single
        parameter set. The most frequent outcome is returned.
        """
        bound_circ = self.circuit.assign_parameters(
            dict(zip(self.circuit.parameters, parameter_set)), inplace=False
        )
        result = self.sampler.run(bound_circ, shots=num_shots)
        return result.get_counts().most_common(1)[0][0]  # most frequent bitstring

__all__ = ["UnifiedSamplerQNN", "FastBaseEstimator", "FastEstimator"]
