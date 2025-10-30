"""Hybrid quantum estimator module.

Provides a FastBaseEstimator capable of evaluating expectation values for a
parametrised quantum circuit.  The implementation supports both Qiskit
and StrawberryFields programs.  A photonic fraud‑detection builder is
included, mirroring the classical counterpart.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Optional, Union

# Qiskit imports
try:
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.quantum_info import Statevector
    from qiskit.quantum_info.operators.base_operator import BaseOperator
except Exception:  # pragma: no cover
    QuantumCircuit = None  # type: ignore[assignment]
    Aer = None  # type: ignore[assignment]
    execute = None  # type: ignore[assignment]
    Statevector = None  # type: ignore[assignment]
    BaseOperator = None  # type: ignore[assignment]

# StrawberryFields imports
try:
    import strawberryfields as sf
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
except Exception:  # pragma: no cover
    sf = None  # type: ignore[assignment]

QuantumCircuitOrProgram = Union[QuantumCircuit, "sf.Program"]  # type: ignore[assignment]


class FastBaseEstimator:
    """Evaluate a parametrised quantum circuit on a list of parameter sets.

    The class works with either a fully parameterised Qiskit ``QuantumCircuit`` or a
    StrawberryFields ``Program``.  Exact expectation values are obtained via a
    state‑vector backend; shot‑level evaluation is not implemented in this
    simplified version but the API allows a ``shots`` argument for future
    extensions.
    """

    def __init__(
        self,
        circuit: QuantumCircuitOrProgram,
        *,
        shots: int | None = None,
        noise: Optional[object] = None,
    ) -> None:
        self.circuit = circuit
        self.shots = shots
        self.noise = noise

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuitOrProgram:
        """Return a circuit with parameters bound to concrete values."""
        if isinstance(self.circuit, QuantumCircuit):
            if len(parameter_values)!= len(self.circuit.parameters):
                raise ValueError("Parameter count mismatch.")
            mapping = {p: v for p, v in zip(self.circuit.parameters, parameter_values)}
            return self.circuit.assign_parameters(mapping, inplace=False)
        elif sf is not None and isinstance(self.circuit, sf.Program):
            if len(parameter_values)!= len(self.circuit.parameters):
                raise ValueError("Parameter count mismatch.")
            mapping = {p: v for p, v in zip(self.circuit.parameters, parameter_values)}
            return self.circuit.assign_parameters(mapping)
        else:
            raise TypeError("Unsupported circuit type.")

    def _evaluate_statevector(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Exact evaluation using a state‑vector simulator."""
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound_circ = self._bind(params)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return expectation values for each observable and parameter set."""
        return self._evaluate_statevector(observables, parameter_sets)


# --------------------------------------------------------------------------- #
# Photonic fraud‑detection program
# --------------------------------------------------------------------------- #

from dataclasses import dataclass


@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic fraud‑detection layer."""

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


def _apply_layer(
    modes: Sequence,
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
    """Apply a photonic layer to the given modes."""
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Construct a StrawberryFields program mirroring the photonic fraud‑detection design."""
    if sf is None:
        raise RuntimeError("StrawberryFields is not available.")
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


__all__ = [
    "FastBaseEstimator",
    "FraudLayerParameters",
    "build_fraud_detection_program",
]
