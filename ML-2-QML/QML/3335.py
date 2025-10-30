"""Quantum implementation of the fraud detection circuit, inspired by photonic and qubit‑based fully connected layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute


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


class FraudDetectionQuantumCircuit:
    """Hybrid quantum circuit that emulates photonic layers with qubit gates and a fully connected quantum sub‑circuit."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters], shots: int = 1024) -> None:
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.input_params = input_params
        self.layers = list(layers)
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        self._apply_layer(qc, self.input_params, clip=False)
        for layer in self.layers:
            self._apply_layer(qc, layer, clip=True)
        qc.measure_all()
        return qc

    def _apply_layer(self, qc: QuantumCircuit, params: FraudLayerParameters, clip: bool) -> None:
        # Beamsplitter analogue
        qc.cx(0, 1)
        theta = params.bs_theta if not clip else _clip(params.bs_theta, 5)
        phi = params.bs_phi if not clip else _clip(params.bs_phi, 5)
        qc.ry(theta, 0)
        qc.rz(phi, 0)
        qc.ry(theta, 1)
        qc.rz(phi, 1)

        # Phases
        for i, phase in enumerate(params.phases):
            qc.rz(phase, i)

        # Squeezing analogue
        for i, (r, phi_s) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_clip = r if not clip else _clip(r, 5)
            qc.rz(r_clip, i)

        # Displacement analogue
        for i, (r, phi_d) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_clip = r if not clip else _clip(r, 5)
            qc.rz(r_clip, i)

        # Kerr analogue
        for i, k in enumerate(params.kerr):
            k_clip = k if not clip else _clip(k, 1)
            qc.rz(k_clip, i)

        # Second beamsplitter analogue
        qc.cx(0, 1)

    def run(self) -> float:
        job = execute(self.circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array(list(counts.keys()), dtype=int)
        expectation = np.sum(states * probs)
        return expectation


class FraudDetectorHybrid:
    """Quantum fraud detection circuit wrapper."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        self.quantum_circuit = FraudDetectionQuantumCircuit(input_params, layers)

    def run(self) -> float:
        return self.quantum_circuit.run()


__all__ = ["FraudLayerParameters", "FraudDetectionQuantumCircuit", "FraudDetectorHybrid"]
