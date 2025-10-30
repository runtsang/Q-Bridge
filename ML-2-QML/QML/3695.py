"""Hybrid quantum self‑attention and fraud‑detection implementation.

The quantum side mirrors the classical architecture: a Qiskit
self‑attention circuit followed by a Strawberry Fields photonic
program.  The interface is intentionally similar to the classical
module to simplify comparative experiments.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable

# --- Quantum self‑attention --------------------------------------
class QuantumSelfAttention:
    """Qiskit implementation of a self‑attention style block."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qreg = QuantumRegister(n_qubits, "q")
        self.creg = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.qreg, self.creg)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qreg, self.creg)
        return qc

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        qc = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)

# --- Fraud‑detection (photonic) ----------------------------------
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

def _apply_layer(modes, params: FraudLayerParameters, *, clip: bool):
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
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

# --- Hybrid quantum model -----------------------------------------
class HybridAttentionFraudDetector:
    """Quantum hybrid of self‑attention and fraud detection.

    The attention block is executed on a Qiskit backend; the fraud
    detection block is executed on a Strawberry Fields Gaussian
    simulator.  The two stages can be chained by passing the
    measurement results of the attention circuit to the photonic
    program via classical post‑processing.
    """
    def __init__(
        self,
        attention_qubits: int,
        fraud_input: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        attention_backend=None,
        fraud_backend=None,
    ):
        self.attention = QuantumSelfAttention(attention_qubits)
        self.attention_backend = attention_backend or qiskit.Aer.get_backend("qasm_simulator")
        self.fraud_prog = build_fraud_detection_program(fraud_input, fraud_layers)
        self.fraud_backend = fraud_backend or sf.Engine("gaussian")

    def run_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        return self.attention.run(
            self.attention_backend,
            rotation_params,
            entangle_params,
            shots=shots,
        )

    def run_fraud(self, inputs: np.ndarray):
        """Execute the photonic fraud‑detection circuit on the provided
        classical data.  `inputs` should be a 2‑dimensional vector
        matching the mode count of the program.
        """
        # Simple encoding: displace the modes by the input amplitudes.
        eng = self.fraud_backend
        eng.run(self.fraud_prog, run_options={"inputs": inputs})
        return eng.state()

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ):
        attn_counts = self.run_attention(rotation_params, entangle_params, shots)
        # Classical post‑processing: map counts to a 2‑dim vector
        # For illustration we take the most frequent outcome.
        most_common = max(attn_counts, key=attn_counts.get)
        encoded = np.array([int(bit) for bit in most_common], dtype=float)
        fraud_state = self.run_fraud(encoded)
        return {"attention_counts": attn_counts, "fraud_state": fraud_state}

__all__ = ["HybridAttentionFraudDetector", "FraudLayerParameters"]
