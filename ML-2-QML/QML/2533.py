"""Quantum‑only implementation of the fraud‑detection circuit.

The class builds a parameterised Qiskit circuit that mirrors the structure of the
photonic model from the seed.  It exposes a ``run`` method that accepts a batch
of parameter vectors and returns the expectation value of the Z‑operator on the
first qubit.

The implementation demonstrates how the same logical layer can be realised
purely quantum‑mechanically while remaining compatible with the classical
anchor.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import Aer, execute
from dataclasses import dataclass
from typing import Iterable, Sequence

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

def _apply_layer(qc: qiskit.QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    # Beam‑splitter analogue: two RZ rotations
    theta = params.bs_theta
    phi = params.bs_phi
    if clip:
        theta = _clip(theta, 5)
        phi = _clip(phi, 5)
    qc.rz(theta, 0)
    qc.rz(phi, 1)

    # Phase shifters
    for i, phase in enumerate(params.phases):
        qc.rz(phase, i)

    # Squeezing → Y‑rotations
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qc.ry(r if not clip else _clip(r, 5), i)

    # Displacement → X‑rotations
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qc.rx(r if not clip else _clip(r, 5), i)

    # Kerr non‑linearity → Z‑rotations
    for i, k in enumerate(params.kerr):
        qc.rz(k if not clip else _clip(k, 1), i)

def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qiskit.QuantumCircuit:
    """Create a Qiskit circuit that mirrors the layered photonic model."""
    qc = qiskit.QuantumCircuit(2)
    _apply_layer(qc, input_params, clip=False)
    for layer in layers:
        _apply_layer(qc, layer, clip=True)
    qc.measure_all()
    return qc

class FraudDetectionHybrid:
    """Quantum‑only fraud‑detection circuit with a convenient run interface."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        backend: qiskit.providers.BaseBackend | None = None,
        shots: int = 1024,
    ) -> None:
        self.circuit = build_fraud_detection_circuit(input_params, layers)
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

    def run(self, parameter_vectors: Iterable[Sequence[float]]) -> np.ndarray:
        """Execute the circuit for each parameter vector and return expectation values."""
        expectations = []
        for vec in parameter_vectors:
            # Bind all circuit parameters in order of appearance
            param_dict = {p: v for p, v in zip(self.circuit.parameters, vec)}
            job = execute(
                self.circuit,
                backend=self.backend,
                shots=self.shots,
                parameter_binds=[param_dict],
            )
            result = job.result()
            counts = result.get_counts(self.circuit)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
            expectations.append(np.sum(states * probs))
        return np.array(expectations)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
