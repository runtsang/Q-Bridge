"""
Hybrid quantum model that mirrors the classical pipeline using Qiskit and Strawberry Fields.
Author: gpt-oss-20b
"""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable

# --------------------------------------------------------------------------- #
# 1. Quantum convolution (quanvolution) ------------------------------------ #
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """
    Variational circuit that emulates a 2‑D convolution via parameterised RX gates
    followed by a random circuit and measurement.
    """
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {theta: (np.pi if val > self.threshold else 0) for theta, val in zip(self.theta, dat)}
            param_binds.append(bind)
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# 2. Quantum self‑attention ----------------------------------------------- #
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """
    Variational circuit implementing a self‑attention style block.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
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
        circuit = self._build(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

# --------------------------------------------------------------------------- #
# 3. Photonic fraud‑detection program ------------------------------------- #
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

def _apply_layer(modes: Iterable, params: FraudLayerParameters, clip: bool) -> None:
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

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                  layers: Iterable[FraudLayerParameters]) -> sf.Program:
    """
    Creates a Strawberry Fields program that mirrors the layered photonic architecture.
    """
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

# --------------------------------------------------------------------------- #
# 4. Hybrid quantum pipeline ----------------------------------------------- #
# --------------------------------------------------------------------------- #
class HybridQuantumConvAttentionFraud:
    """
    End‑to‑end quantum model: quanvolution → quantum self‑attention → photonic fraud detection.
    """
    def __init__(self,
                 conv_kernel: int = 2,
                 conv_thresh: float = 127,
                 attn_qubits: int = 4,
                 fraud_input_params: FraudLayerParameters | None = None,
                 fraud_layer_params: Iterable[FraudLayerParameters] | None = None):
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(conv_kernel, self.backend, shots=100, threshold=conv_thresh)
        self.attn = QuantumSelfAttention(attn_qubits)
        if fraud_input_params is None:
            fraud_input_params = FraudLayerParameters(
                bs_theta=0.0, bs_phi=0.0, phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0), squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0), displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0))
        if fraud_layer_params is None:
            fraud_layer_params = []
        self.fraud_prog = build_fraud_detection_program(fraud_input_params, fraud_layer_params)
        self.sf_backend = sf.Simulator()

    def run(self, data):
        # 1. Quanvolution
        conv_out = self.conv.run(data)  # scalar probability

        # 2. Quantum self‑attention (dummy parameters derived from conv_out)
        rotation_params = np.full(self.attn.n_qubits * 3, conv_out)
        entangle_params  = np.full(self.attn.n_qubits - 1, conv_out)
        attn_counts = self.attn.run(self.backend, rotation_params, entangle_params, shots=1024)

        # 3. Convert measurement counts into a simple vector
        attn_vector = np.array([sum(int(b) for b in key) * val
                                for key, val in attn_counts.items()]) / (1024 * self.attn.n_qubits)

        # 4. Photonic fraud detection (use the first two components as displacement amplitudes)
        args = dict(displacement=[attn_vector[0], attn_vector[1]],
                    displacement_phase=[0.0, 0.0])
        job = self.sf_backend.run(self.fraud_prog, args=args)
        return job.results.samples
