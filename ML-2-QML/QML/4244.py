from __future__ import annotations

from typing import Iterable, List, Sequence
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import StatevectorSampler as Sampler

class HybridSamplerQNN:
    """
    Quantum implementation of the hybrid sampler.
    Builds a compact variational circuit that mirrors the classical
    QCNN‑style architecture and supports batched evaluation with
    optional shot noise.
    """
    def __init__(self) -> None:
        # Feature map that encodes 2‑dim input into 2 qubits
        feature_map = ZFeatureMap(2)
        # Ansatz that emulates a convolution + pooling block
        ansatz = self._build_ansatz()
        # Compose feature map and ansatz into a single circuit
        self._circuit = ansatz.compose(feature_map)
        # Statevector sampler primitive
        self._sampler = Sampler()

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        # Convolution‑like block
        qc.ry(ParameterVector("θ0", 1)[0], 0)
        qc.ry(ParameterVector("θ1", 1)[0], 1)
        qc.cx(0, 1)
        # Pooling‑like block
        qc.rz(ParameterVector("θ2", 1)[0], 1)
        qc.cx(1, 0)
        # Weight parameters
        qc.ry(ParameterVector("w0", 1)[0], 0)
        qc.ry(ParameterVector("w1", 1)[0], 1)
        return qc

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        all_params = list(self._circuit.parameters)
        if len(parameter_values)!= len(all_params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(all_params, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Evaluate expectation values of observables for each set of parameters.
        Supports optional Gaussian shot noise.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                rng.normal(complex(val).real, max(1e-6, 1 / shots))
                + 1j * rng.normal(complex(val).imag, max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

def SamplerQNN() -> HybridSamplerQNN:
    """
    Factory function that returns an instance of the quantum hybrid sampler.
    """
    return HybridSamplerQNN()

__all__ = ["HybridSamplerQNN", "SamplerQNN"]
