"""Hybrid photonic‑quantum fraud‑detection circuit with a Qiskit self‑attention block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer


@dataclass
class FraudLayerParameters:
    """Parameters for one photonic‑style layer."""
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


def _apply_layer(q: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Photonic program that implements the fraud‑detection layers."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog


def _build_attention_circuit(rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
    """Simple Qiskit self‑attention circuit used as a post‑processing block."""
    n = len(rotation_params) // 3
    qr = QuantumRegister(n, "q")
    cr = ClassicalRegister(n, "c")
    circ = QuantumCircuit(qr, cr)
    for i in range(n):
        circ.rx(rotation_params[3 * i], i)
        circ.ry(rotation_params[3 * i + 1], i)
        circ.rz(rotation_params[3 * i + 2], i)
    for i in range(n - 1):
        circ.crx(entangle_params[i], i, i + 1)
    circ.measure(qr, cr)
    return circ


class FraudDetectionAttention:
    """Hybrid interface that runs the photonic fraud circuit and a Qiskit attention block."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_params: Iterable[FraudLayerParameters],
        attention_rot: np.ndarray,
        attention_ent: np.ndarray,
        backend=None,
    ) -> None:
        self.prog = build_fraud_detection_program(input_params, hidden_params)
        self.attention_circ = _build_attention_circuit(attention_rot, attention_ent)
        self.backend = backend or Aer.get_backend("qasm_simulator")

    def run(self, inputs: np.ndarray) -> dict[str, int]:
        """Execute the photonic program (placeholder) and the attention circuit."""
        # For illustration we only run the quantum attention circuit.
        job = execute(self.attention_circ, self.backend, shots=1024)
        return job.result().get_counts(self.attention_circ)


__all__ = ["FraudLayerParameters", "FraudDetectionAttention"]
