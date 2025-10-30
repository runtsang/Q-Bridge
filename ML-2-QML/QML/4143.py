from __future__ import annotations

import numpy as np
from typing import Iterable, Sequence, List, Union
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
import torchquantum as tq
import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class FraudLayerParams:
    """Parameters for a fraud‑detection style quantum layer."""
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


def build_fraud_detection_quantum_program(
    input_params: FraudLayerParams,
    layers: Sequence[FraudLayerParams],
) -> QuantumCircuit:
    circ = QuantumCircuit(2)

    def apply_layer(params: FraudLayerParams, clip: bool):
        theta, phi = params.bs_theta, params.bs_phi
        circ.rx(theta, 0)
        circ.rx(phi, 1)
        for phase in params.phases:
            circ.rz(phase, 0)
            circ.rz(phase, 1)
        for r, p in zip(params.squeeze_r, params.squeeze_phi):
            val = _clip(r, 5) if clip else r
            circ.ry(val, 0)
            circ.ry(val, 1)
        for r, p in zip(params.displacement_r, params.displacement_phi):
            val = _clip(r, 5) if clip else r
            circ.cx(0, 1)
        for k in params.kerr:
            val = _clip(k, 1) if clip else k
            circ.rz(val, 0)

    apply_layer(input_params, clip=False)
    for lay in layers:
        apply_layer(lay, clip=True)
    return circ


class FastHybridEstimator:
    """Quantum‑centric estimator that handles pure circuits or hybrid torchquantum modules."""
    def __init__(self, model: Union[QuantumCircuit, tq.QuantumModule]) -> None:
        self.model = model
        self._is_hybrid = isinstance(model, tq.QuantumModule)

    def evaluate(
        self,
        observables: Iterable[Union[BaseOperator, object]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        if self._is_hybrid:
            return self._eval_hybrid(observables, parameter_sets, shots, seed)
        else:
            return self._eval_circuit(observables, parameter_sets, shots, seed)

    def _eval_circuit(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            circ = self._bind(params)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return self._add_shots(results, shots, seed)

    def _eval_hybrid(
        self,
        observables: Iterable[object],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> List[List[complex]]:
        raw = self.model.evaluate(observables, parameter_sets)
        return self._add_shots(raw, shots, seed)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self.model.parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.model.parameters, values))
        return self.model.assign_parameters(mapping, inplace=False)

    @staticmethod
    def _add_shots(
        raw: List[List[complex]], shots: int | None, seed: int | None
    ) -> List[List[complex]]:
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [
                rng.normal(complex(val.real, val.imag), 1 / np.sqrt(shots)) for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastHybridEstimator", "FraudLayerParams", "build_fraud_detection_quantum_program"]
