"""Combined quantum fraud detection with quanvolution and photonic layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import execute, Aer
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


# --- Quantum convolution (from Conv.py) --------------------------------

class QuanvCircuit:
    """Quantum filter implemented with Qiskit."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the circuit on a 2‑D kernel and return average |1> probability."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


# --- Fraud detection photonic layers ------------------------------------

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


# --- Hybrid quantum model ----------------------------------------------

class FraudDetectorHybrid:
    """
    Quantum fraud detector that first applies a quanvolution circuit
    and then processes the result through a photonic fraud detection program.
    """
    def __init__(
        self,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
        conv_shots: int = 100,
        fraud_input_params: FraudLayerParameters | None = None,
        fraud_layers: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        backend = Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(conv_kernel_size, backend, conv_shots, conv_threshold)

        if fraud_input_params is None:
            fraud_input_params = FraudLayerParameters(
                bs_theta=0.5,
                bs_phi=0.3,
                phases=(0.1, -0.1),
                squeeze_r=(0.2, 0.2),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        if fraud_layers is None:
            fraud_layers = []

        self.fraud_program = build_fraud_detection_program(fraud_input_params, fraud_layers)
        self.sampler = sf.backends.Simulator()

    def run(self, image: np.ndarray) -> float:
        """
        Forward pass: image → quanvolution → photonic fraud detection → prediction.
        """
        conv_out = self.conv.run(image)
        # Use the classical output to set displacement parameters in a new program
        displaced_params = FraudLayerParameters(
            bs_theta=0.5,
            bs_phi=0.3,
            phases=(0.1, -0.1),
            squeeze_r=(0.2, 0.2),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(conv_out, conv_out),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        program = build_fraud_detection_program(displaced_params, [])
        result = self.sampler.run(program)
        return result.samples[:, 0].mean()


__all__ = ["FraudDetectorHybrid"]
