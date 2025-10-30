from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

from dataclasses import dataclass
from typing import Iterable, Sequence

# Quantum convolution filter similar to the QML Conv reference
class QuantumConvFilter:
    def __init__(self, kernel_size: int, threshold: float, shots: int = 100):
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0 for i, val in enumerate(dat)}
            param_binds.append(bind)
        job = qiskit.execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# Photonic layer parameters
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

class FraudDetectionHybrid:
    """
    Quantum‑classical hybrid model that first applies a quantum convolution
    filter (via Qiskit) to the input image, encodes the resulting probability
    into Strawberry Fields modes, and then propagates the state through
    a photonic‑like layered network.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        kernel_size: int = 2,
        threshold: float = 0.0,
        shots: int = 100,
    ) -> None:
        self.quantum_filter = QuantumConvFilter(kernel_size, threshold, shots)
        self.input_params = input_params
        self.layers = list(layers)

    def run(self, data: np.ndarray) -> float:
        """
        Execute the hybrid circuit and return the mean photon number
        measured across the two modes – a proxy for fraud risk.
        """
        # Quantum convolution step
        p = self.quantum_filter.run(data)

        # Build a fresh Program each call to embed the current probability
        prog = sf.Program(2)
        with prog.context as q:
            # Encode the conv probability into mode 0
            Dgate(p) | q[0]
            # Apply the photonic layers
            _apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                _apply_layer(q, layer, clip=True)

        # Run the program on a simulator
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 8})
        result = eng.run(prog, shots=1000)
        # Compute mean photon number across both modes
        mean_photon = np.mean([sum(result.samples[:, i]) / 1000 for i in range(2)])
        return mean_photon

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    kernel_size: int = 2,
    threshold: float = 0.0,
    shots: int = 100,
) -> FraudDetectionHybrid:
    """Convenience constructor for the hybrid quantum‑classical fraud detection model."""
    return FraudDetectionHybrid(input_params, layers, kernel_size, threshold, shots)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
