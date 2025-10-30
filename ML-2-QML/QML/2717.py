"""Hybrid quantum fraud detection program combining a Strawberry Fields photonic circuit
and a Qiskit sampler circuit.  The class exposes methods to generate both sub‑circuits
and to execute them on simulators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as QiskitSampler


@dataclass
class FraudLayerParams:
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


def _apply_photonic_layer(q: Sequence, params: FraudLayerParams, clip: bool = False) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | q[i]
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | q[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | q[i]


def build_photonic_program(
    input_params: FraudLayerParams,
    hidden_params: Iterable[FraudLayerParams],
) -> sf.Program:
    prog = sf.Program(2)
    with prog.context as q:
        _apply_photonic_layer(q, input_params, clip=False)
        for p in hidden_params:
            _apply_photonic_layer(q, p, clip=True)
    return prog


def build_sampler_circuit() -> QuantumCircuit:
    """Construct a Qiskit circuit that implements the SamplerQNN."""
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    for i in range(4):
        qc.ry(weights[i], i % 2)
    qc.cx(0, 1)
    return qc


class FraudDetectionHybridQuantum:
    """Quantum counterpart of FraudDetectionHybrid.

    Provides both the photonic program and the Qiskit sampler circuit.
    The class can execute them on simulators or hybrid devices.
    """
    def __init__(
        self,
        input_params: FraudLayerParams,
        hidden_params: Iterable[FraudLayerParams],
    ) -> None:
        self.photonic_prog = build_photonic_program(input_params, hidden_params)
        self.sampler_qc = build_sampler_circuit()
        self.sampler = QiskitSampler()

    def run_photonic(self, backend: sf.backends.Backend) -> sf.results.Result:
        """Execute the photonic program on a Strawberry Fields backend."""
        return backend.run(self.photonic_prog)

    def run_sampler(self, bindings: dict) -> dict:
        """Execute the sampler circuit with given parameter bindings."""
        return self.sampler.run(self.sampler_qc, parameters=bindings)


__all__ = ["FraudLayerParams", "FraudDetectionHybridQuantum"]
