"""
FraudDetectionHybrid – a quantum model that interleaves Strawberry Fields photonic gates with a Qiskit parameterised circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter


@dataclass
class FraudLayerParameters:
    """
    Parameters describing a single photonic‑style layer.
    """
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


class QiskitFCL:
    """
    Simple parameterised Qiskit circuit that emulates the fully‑connected layer.
    The circuit applies a single Ry rotation per qubit and measures the expectation
    of the computational basis state.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 1000) -> None:
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = Parameter("θ")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()), dtype=float)
        states = np.array([int(k, 2) for k in result.keys()], dtype=float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])


class FraudDetectionQML:
    """
    Hybrid quantum fraud‑detection circuit.
    The program consists of:
        1. A Strawberry Fields photonic layer per fraud layer.
        2. A Qiskit FCL circuit that processes a set of angles.
    The overall expectation is the sum of photonic measurement and Qiskit expectation.
    """
    def __init__(self,
                 input_params: FraudLayerParameters,
                 hidden_params: Sequence[FraudLayerParameters],
                 fcl_qubits: int = 1,
                 fcl_shots: int = 1000) -> None:
        self.program = sf.Program(2)
        with self.program.context as q:
            self._apply_layer(q, input_params, clip=False)
            for p in hidden_params:
                self._apply_layer(q, p, clip=True)
        self.fcl = QiskitFCL(n_qubits=fcl_qubits, shots=fcl_shots)

    def _apply_layer(self, modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
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

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the Strawberry Fields program on the Gaussian simulator
        and add the expectation from the Qiskit FCL circuit.
        """
        eng = sf.Engine("gaussian")
        results = eng.run(self.program, shots=1000)
        photonic_expect = results.samples.mean(axis=0).sum()
        qiskit_expect = self.fcl.run(thetas)
        return photonic_expect + qiskit_expect


__all__ = ["FraudLayerParameters", "QiskitFCL", "FraudDetectionQML"]
