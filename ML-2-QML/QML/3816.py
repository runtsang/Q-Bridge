"""Hybrid fraud‑detection quantum model that fuses a Qiskit quantum convolution
with a Strawberry Fields photonic program.

The implementation mirrors the classical design but replaces the
classical convolution by a parameter‑ised quantum circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


def _apply_layer(modes: List, params: FraudLayerParameters, *, clip: bool) -> None:
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


class QuantumConvCircuit:
    """
    Quantum convolution implemented with Qiskit.
    Each patch of the input image is encoded into rotation angles
    and processed by a fixed random circuit.
    """

    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
        ]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the circuit on a single kernel patch."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
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


class FraudDetectionHybrid:
    """
    Quantum‑classical hybrid fraud‑detection model.

    The first stage applies a Qiskit quantum convolution over sliding
    kernels of the input image.  The resulting expectation values are
    injected as displacements into a Strawberry Fields photonic program
    built from the supplied `FraudLayerParameters`.
    """

    def __init__(
        self,
        conv_kernel_size: int = 2,
        conv_threshold: float = 127,
        fraud_params: List[FraudLayerParameters] | None = None,
        shots: int = 100,
        backend=None,
    ) -> None:
        self.conv = QuantumConvCircuit(
            conv_kernel_size,
            backend or qiskit.Aer.get_backend("qasm_simulator"),
            shots,
            conv_threshold,
        )
        self.fraud_params = fraud_params or []
        if self.fraud_params:
            self.photonic_prog = build_fraud_detection_program(
                self.fraud_params[0], self.fraud_params[1:]
            )
        else:
            self.photonic_prog = None

    def _extract_patches(self, data: np.ndarray) -> List[np.ndarray]:
        """Yield overlapping 2×2 patches from a 2‑D image."""
        h, w = data.shape
        patches = []
        for i in range(h - 1):
            for j in range(w - 1):
                patches.append(data[i : i + 2, j : j + 2])
        return patches

    def run(self, data: np.ndarray) -> float:
        """
        Execute the hybrid model on a single 2‑D input.

        Parameters
        ----------
        data : np.ndarray
            Shape (H, W) image.  Values should be in [0, 255] for the
            default threshold.

        Returns
        -------
        float
            Fraud‑risk score in the interval [0, 1].
        """
        patches = self._extract_patches(data)
        conv_out = np.array([self.conv.run(p) for p in patches])

        if self.photonic_prog is None:
            return float(np.mean(conv_out))

        # Inject the convolution output into the photonic program as
        # displacements on the two modes.
        prog = sf.Program(2)
        with prog.context as q:
            # use the first two values of conv_out; pad or truncate if needed
            disp = conv_out[:2]
            disp = np.pad(disp, (0, 2 - len(disp)), "constant")
            for i, val in enumerate(disp):
                Dgate(val) | q[i]
            _apply_layer(q, self.fraud_params[0], clip=False)
            for layer in self.fraud_params[1:]:
                _apply_layer(q, layer, clip=True)

        eng = sf.Engine("gaussian")
        result = eng.run(prog)
        # Expectation value of the photon number in mode 0
        return float(result.expectation_values[0][0])

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
