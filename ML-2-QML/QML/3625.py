from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import execute
import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.backends import FockBackend
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

class HybridConvFraud:
    """Quantum hybrid model combining a quanvolution circuit with a photonic fraud‑detection
    program.  The quantum convolution produces a scalar feature that can be used to bias the
    photonic sub‑circuit.  The full pipeline demonstrates how a classical image patch can be encoded,
    processed quantumly, and fed into a continuous‑variable machine learning block.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the square image patch to be processed by the quanvolution circuit.
    conv_shots : int, optional
        Number of shots for the Qiskit simulation.  Default is 100.
    conv_threshold : int, optional
        Threshold used when mapping pixel intensities to rotation angles.  Default is 127.
    sf_backend : sf.backends.Backend, optional
        Strawberry‑Fields backend; defaults to the Fock backend.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_shots: int = 100,
        conv_threshold: int = 127,
        sf_backend=None,
    ) -> None:
        self.kernel_size = kernel_size
        self.conv_shots = conv_shots
        self.conv_threshold = conv_threshold
        self.conv_backend = qiskit.Aer.get_backend("qasm_simulator")
        self.sf_backend = sf_backend or FockBackend()
        self._circuit, self.theta = self._build_conv_circuit()

    def _build_conv_circuit(self):
        n_qubits = self.kernel_size ** 2
        qc = qiskit.QuantumCircuit(n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n_qubits, 2)
        qc.measure_all()
        return qc, theta

    def run_conv(self, data: np.ndarray) -> float:
        data_flat = np.reshape(data, (1, self.kernel_size ** 2))
        param_binds = []
        for sample in data_flat:
            bind = {theta: np.pi if val > self.conv_threshold else 0 for theta, val in zip(self.theta, sample)}
            param_binds.append(bind)
        job = execute(self._circuit, self.conv_backend, shots=self.conv_shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.conv_shots * self.kernel_size ** 2)

    def build_fraud_program(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> sf.Program:
        program = build_fraud_detection_program(input_params, layers)
        program.backend = self.sf_backend
        return program

    def run_fraud(self, program: sf.Program) -> float:
        eng = Engine(program.backend)
        eng.run(program)
        state = eng.state
        mean_photon = sum([state.expectation_value(sf.ops.Number(n)) for n in range(program.num_modes)])
        return mean_photon

    def run(self, data: np.ndarray, fraud_params: Iterable[FraudLayerParameters]) -> tuple[float, float]:
        conv_out = self.run_conv(data)
        fraud_input = list(fraud_params)
        input_params = fraud_input[0]
        layers = fraud_input[1:]
        program = self.build_fraud_program(input_params, layers)
        fraud_out = self.run_fraud(program)
        return conv_out, fraud_out

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "HybridConvFraud"]
