"""Hybrid fast estimator with Qiskit backend and photonic fraud‑detection program builder."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import BaseOperator, Statevector
from qiskit import Aer, execute

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(circuit: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    # Simple mapping of photonic parameters to standard gates for illustration
    circuit.rx(params.bs_theta, 0)
    circuit.rz(params.bs_phi, 0)
    circuit.rx(params.bs_theta, 1)
    circuit.rz(params.bs_phi, 1)

    for i, phase in enumerate(params.phases):
        circuit.rz(phase, i)

    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circuit.rz(_clip(r, 5), i)

    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circuit.rx(_clip(r, 5), i)

    for i, k in enumerate(params.kerr):
        circuit.rz(_clip(k, 1), i)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Create a Qiskit circuit that mimics the photonic fraud‑detection architecture."""
    qc = QuantumCircuit(2)
    _apply_layer(qc, input_params, clip=False)
    for layer in layers:
        _apply_layer(qc, layer, clip=True)
    return qc


class FastBaseEstimator:
    """Evaluate a Qiskit circuit for multiple parameter sets and observables with optional shot noise."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circ = self._bind(values)
            if shots is None:
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                backend = Aer.get_backend("qasm_simulator")
                job = execute(bound_circ, backend=backend, shots=shots, seed_simulator=seed)
                result = job.result()
                counts = result.get_counts()
                row = []
                for obs in observables:
                    exp = 0.0
                    for state, freq in counts.items():
                        # Simplified eigenvalue mapping: +1 for even number of 1s, -1 otherwise
                        eig = 1 if state.count("1") % 2 == 0 else -1
                        exp += eig * freq / shots
                    row.append(complex(exp))
            results.append(row)
        return results


__all__ = ["FastBaseEstimator", "FraudLayerParameters", "build_fraud_detection_program"]
