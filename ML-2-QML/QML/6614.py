from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from strawberryfields.ops import MeasureFock
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


@dataclass
class AttentionParameters:
    """Parameters for a quantum self‑attention block."""
    rotation_params: np.ndarray
    entangle_params: np.ndarray


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


class QuantumSelfAttention:
    """Quantum self‑attention circuit built with Qiskit."""
    def __init__(self, n_qubits: int):
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
            circuit.cx(self.qr[i], self.qr[i + 1])
            circuit.rz(entangle_params[i]) | (self.qr[i], self.qr[i + 1])
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        probs = np.array(list(counts.values())) / shots
        return probs


class FraudDetectionQML:
    """Hybrid quantum‑classical fraud‑detection model that uses a Qiskit
    self‑attention block to generate displacement parameters for a
    Strawberry Fields photonic program."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        attention_params: AttentionParameters,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.attention_params = attention_params
        self.attention = QuantumSelfAttention(n_qubits=4)
        self.backend = Aer.get_backend("qasm_simulator")

    def _displacement_from_attention(self, probs: np.ndarray) -> tuple[float, float]:
        r1 = 2 * probs[0] if probs.size > 0 else 0.0
        r2 = 2 * probs[1] if probs.size > 1 else 0.0
        return (r1, r2)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        outputs = []
        for x in inputs:
            probs = self.attention.run(
                self.backend,
                self.attention_params.rotation_params,
                self.attention_params.entangle_params,
            )
            disp_r = self._displacement_from_attention(probs)
            program = sf.Program(2)
            with program.context as q:
                _apply_layer(q, self.input_params, clip=False)
                disp_params = FraudLayerParameters(
                    bs_theta=self.input_params.bs_theta,
                    bs_phi=self.input_params.bs_phi,
                    phases=self.input_params.phases,
                    squeeze_r=self.input_params.squeeze_r,
                    squeeze_phi=self.input_params.squeeze_phi,
                    displacement_r=disp_r,
                    displacement_phi=self.input_params.displacement_phi,
                    kerr=self.input_params.kerr,
                )
                _apply_layer(q, disp_params, clip=True)
                for layer in self.layers[1:]:
                    _apply_layer(q, layer, clip=True)
                MeasureFock(0) | q[0]
                MeasureFock(1) | q[1]
            eng = sf.Engine("tf", backend_options={"device": "CPU"})
            result = eng.run(program).measurements
            out = np.sum(result)
            outputs.append(out)
        return np.array(outputs)

def build_fraud_detection_qml(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    attention_params: AttentionParameters,
) -> FraudDetectionQML:
    """Convenience constructor for the hybrid QML model."""
    return FraudDetectionQML(input_params, layers, attention_params)
