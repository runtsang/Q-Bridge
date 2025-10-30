"""Hybrid fraud detection module using Strawberry Fields and Qiskit self‑attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# --------------------------------------------------------------------------- #
#  Photonic‑style layer definition (quantum)
# --------------------------------------------------------------------------- #
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

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

# --------------------------------------------------------------------------- #
#  Quantum self‑attention module
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Self‑attention style block implemented with Qiskit."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

# --------------------------------------------------------------------------- #
#  Hybrid model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid:
    """
    Executes a photonic fraud‑detection circuit and a quantum self‑attention block.
    The outputs of both are concatenated and passed to a classical read‑out,
    demonstrating a hybrid inference pipeline.
    """

    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        attention_params: tuple[np.ndarray, np.ndarray],
        backend=None,
    ):
        self.program = build_fraud_detection_program(fraud_params, fraud_layers)
        self.attention = QuantumSelfAttention(n_qubits=attention_params[0].shape[0] // 3)
        self.rotation_params, self.entangle_params = attention_params
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

    def run(self, inputs: np.ndarray, shots: int = 1024) -> dict:
        """Execute both photonic and quantum parts and return combined measurement counts."""
        # Photonic simulation
        eng = sf.Engine("fock", backend="gaussian", shots=shots)
        results = eng.run(self.program, args={"input_state": inputs})
        photonic_counts = results.get_counts()

        # Quantum self‑attention
        quantum_counts = self.attention.run(
            self.backend,
            self.rotation_params,
            self.entangle_params,
            shots=shots,
        )

        return {"photonic": photonic_counts, "quantum_attention": quantum_counts}

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "QuantumSelfAttention",
    "FraudDetectionHybrid",
]
