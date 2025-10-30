"""Hybrid quantum‑classical fraud detection model.

The module first executes a quantum self‑attention circuit implemented in Qiskit
to produce a probability distribution over four qubits.  The resulting
distribution is reshaped into a 2‑dimensional displacement vector that
serves as the input state for a Strawberry Fields photonic program
representing the fraud‑detection layers.  The photonic program is
defined by the same ``FraudLayerParameters`` schema as the classical
counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

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

class QuantumSelfAttention:
    """Quantum self‑attention circuit producing a 4‑qubit state."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

def _attention_to_displacement(counts: dict) -> tuple[float, float]:
    """Map measurement counts to a 2‑dimensional displacement vector."""
    total = sum(counts.values())
    probs = np.array([counts.get(f"{i:04b}", 0) for i in range(16)]) / total
    # Collapse to two dimensions via simple linear mapping
    disp = probs.reshape(4, 4).sum(axis=0)
    return float(disp[0] - disp[1]), float(disp[2] - disp[3])

def build_fraud_detection_quantum_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    rotation_params: np.ndarray,
    entangle_params: np.ndarray,
) -> sf.Program:
    """Create a hybrid program: quantum self‑attention + photonic fraud detection."""
    backend = Aer.get_backend("qasm_simulator")
    q_attn = QuantumSelfAttention()
    counts = q_attn.run(backend, rotation_params, entangle_params)
    disp_r, disp_phi = _attention_to_displacement(counts)

    # Override the displacement parameters of the first layer with attention output
    input_params.displacement_r = (disp_r, disp_phi)
    input_params.displacement_phi = (0.0, 0.0)

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

class FraudDetectionHybridAttentionQuantum:
    """Wrapper exposing the hybrid quantum‑classical program."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ):
        self.program = build_fraud_detection_quantum_program(
            input_params, layers, rotation_params, entangle_params
        )

    def get_program(self) -> sf.Program:
        return self.program

__all__ = ["FraudLayerParameters", "QuantumSelfAttention", "build_fraud_detection_quantum_program", "FraudDetectionHybridAttentionQuantum"]
