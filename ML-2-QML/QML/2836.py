"""Hybrid quantum self‑attention and fraud detection framework."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
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

class SelfAttention:
    """Hybrid quantum self‑attention and fraud detection model."""
    def __init__(
        self,
        n_qubits: int = 4,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
    ):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.fraud_params = fraud_params
        self.fraud_program = None
        if fraud_params is not None:
            self.fraud_program = self._build_fraud_program(fraud_params)

    def _build_attention_circuit(
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

    def run_quantum_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circuit = self._build_attention_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

    def _build_fraud_program(self, layers: Iterable[FraudLayerParameters]) -> sf.Program:
        program = sf.Program(2)
        with program.context as q:
            for layer in layers:
                _apply_layer(q, layer, clip=True)
        return program

    def run_quantum_fraud(self, shots: int = 1024):
        if self.fraud_program is None:
            raise RuntimeError("Fraud program not configured.")
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})
        result = eng.run(self.fraud_program, shots=shots)
        return result.samples

    def __repr__(self) -> str:
        return f"<SelfAttention n_qubits={self.n_qubits}>"

__all__ = ["SelfAttention", "FraudLayerParameters"]
