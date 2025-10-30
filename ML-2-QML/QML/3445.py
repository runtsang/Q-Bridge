"""
Hybrid Base Estimator for quantum circuits with shot‑noise simulation
and a factory to build a photonic fraud‑detection program.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    def __init__(
        self,
        bs_theta: float,
        bs_phi: float,
        phases: Tuple[float, float],
        squeeze_r: Tuple[float, float],
        squeeze_phi: Tuple[float, float],
        displacement_r: Tuple[float, float],
        displacement_phi: Tuple[float, float],
        kerr: Tuple[float, float],
    ):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class HybridBaseEstimator:
    """Evaluate a parameterised quantum circuit for expectation values,
    with optional shot‑noise simulation."""
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
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                complex(
                    rng.normal(np.real(v), 1 / shots)
                    + 1j * rng.normal(np.imag(v), 1 / shots)
                )
                for v in row
            ]
            noisy.append(noisy_row)
        return noisy

    @staticmethod
    def build_fraud_detection_program(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> sf.Program:
        """Create a Strawberry‑Fields program that mirrors the classical fraud‑detection network."""
        program = sf.Program(2)

        def _apply_layer(modes, params, clip):
            BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
            for i, phase in enumerate(params.phases):
                Rgate(phase) | modes[i]
            for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
                Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
            BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
            for i, phase in enumerate(params.phases):
                Rgate(phase) | modes[i]
            for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
                Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
            for i, k in enumerate(params.kerr):
                Kgate(k if not clip else _clip(k, 1)) | modes[i]

        with program.context as q:
            _apply_layer(q, input_params, clip=False)
            for layer in layers:
                _apply_layer(q, layer, clip=True)
        return program

__all__ = ["HybridBaseEstimator", "FraudLayerParameters"]
