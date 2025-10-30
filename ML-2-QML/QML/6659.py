"""Hybrid estimator for quantum circuit evaluation with fraud‑detection support.

This module extends the original FastBaseEstimator to work with either a
Qiskit ``QuantumCircuit`` or a Strawberry Fields ``Program``.  The
``HybridEstimator`` can bind parameters, run simulations, and compute
expectation values for a list of observables.  It also exposes a helper
to build a fraud‑detection circuit using the same parameter format as
the classical counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Union

import numpy as np
import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Callable[[Statevector], complex | float]


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


def _apply_layer(
    modes: Sequence,
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
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


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: Union[QuantumCircuit, sf.Program]) -> None:
        self._circuit = circuit
        if isinstance(circuit, QuantumCircuit):
            self._parameters = list(circuit.parameters)
        else:
            # sf.Program has no direct parameter list; we bind at evaluation time
            self._parameters = []

    def _bind_qiskit(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator | sf.operators.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        rng = np.random.default_rng(seed)

        for values in parameter_sets:
            # --- Qiskit path ------------------------------------------------
            if isinstance(self._circuit, QuantumCircuit):
                state = Statevector.from_instruction(self._bind_qiskit(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
                continue

            # --- Strawberry Fields path ------------------------------------
            if isinstance(self._circuit, sf.Program):
                prog = self._circuit.copy()
                if values:
                    prog.bind(values)
                engine = Engine("gaussian")
                result = engine.run(prog)
                state = result.statevector
                # For SF, the observable must support expectation_value on StateVector
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
                continue

            raise TypeError("Unsupported circuit type.")

        # Add optional shot noise to emulate quantum sampling
        if shots is not None:
            noisy_results: List[List[complex]] = []
            for row in results:
                noisy_row = [
                    rng.normal(complex(val.real, val.imag), max(1e-6, 1 / shots))
                    for val in row
                ]
                noisy_results.append(noisy_row)
            return noisy_results

        return results


class HybridEstimator(FastBaseEstimator):
    """Unified estimator that can handle Qiskit or SF circuits and supports fraud‑detection builds."""
    def __init__(
        self,
        circuit: Union[QuantumCircuit, sf.Program],
        *,
        input_params: FraudLayerParameters | None = None,
        layers: Sequence[FraudLayerParameters] | None = None,
    ) -> None:
        if isinstance(circuit, sf.Program) and (input_params is not None or layers is not None):
            # Build a new program from the fraud‑detection spec
            if input_params is None or layers is None:
                raise ValueError("Both input_params and layers must be provided to build a fraud‑detection program.")
            circuit = build_fraud_detection_program(input_params, layers)
        super().__init__(circuit)

    @staticmethod
    def build_fraud_detection_program(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> sf.Program:
        """Convenience wrapper to construct a fraud‑detection SF program."""
        return build_fraud_detection_program(input_params, layers)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FastBaseEstimator",
    "HybridEstimator",
]
