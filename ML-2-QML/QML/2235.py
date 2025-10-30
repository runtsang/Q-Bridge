"""Combined fast estimator for quantum circuits and fraud‑detection photonic programs.

The class evaluates expectation values of a Qiskit circuit for a set of
parameter values.  It supports shot‑noise simulation, parameter binding,
and a factory for building Strawberry Fields photonic fraud‑detection
programs.  The API matches the classical counterpart to ease switching
between regimes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# Optional import for photonic fraud‑detection program
try:
    import strawberryfields as sf
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
except Exception:
    sf = None  # type: ignore[assignment]

@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
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
) -> "sf.Program | None":
    """Create a Strawberry Fields program for the hybrid fraud‑detection model."""
    if sf is None:
        raise RuntimeError("Strawberry Fields is not available.")
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

class FastBaseEstimatorGen034:
    """Evaluate a Qiskit circuit for a set of parameters and observables.

    Parameters
    ----------
    circuit
        A ``qiskit.circuit.QuantumCircuit`` with symbolic parameters.
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
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return expectation values for each parameter set and observable.

        If ``shots`` is provided, Gaussian shot noise with variance ``1/shots``
        is added to each expectation value.
        """
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
            noisy_row = [rng.normal(complex(val.real, 0).real, max(1e-6, 1 / shots))
                         + 1j * rng.normal(complex(val.imag, 0).real, max(1e-6, 1 / shots))
                         for val in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FastBaseEstimatorGen034", "FraudLayerParameters", "build_fraud_detection_program"]
