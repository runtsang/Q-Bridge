from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.primitives import Sampler
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

def _build_layer_circuit(params: FraudLayerParameters, *, clip: bool) -> QuantumCircuit:
    q = QuantumCircuit(2)
    q.ry(params.bs_theta, 0)
    q.ry(params.bs_phi, 1)
    q.cx(0, 1)
    for i, phase in enumerate(params.phases):
        q.rz(phase, i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        q.rz(_clip(r, 5.0), i)
        q.rz(_clip(phi, 5.0), i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        q.ry(_clip(r, 5.0), i)
        q.rz(_clip(phi, 5.0), i)
    for i, k in enumerate(params.kerr):
        q.rz(_clip(k, 1.0), i)
    return q

class FraudDetectionHybrid:
    """
    Quantum implementation of the fraud‑detection architecture.
    The circuit is constructed from photonic‑inspired layers and
    is evaluated via a quantum sampler, producing a probability
    distribution that mirrors the classical sampler network.
    """
    def __init__(self, input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters]) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.sampler = Sampler()

    def _build_full_circuit(self, x: np.ndarray) -> QuantumCircuit:
        q = QuantumCircuit(2)
        q.ry(x[0], 0)
        q.ry(x[1], 1)
        q.compose(_build_layer_circuit(self.input_params, clip=False), inplace=True)
        for layer in self.layers:
            q.compose(_build_layer_circuit(layer, clip=True), inplace=True)
        q.measure_all()
        return q

    def forward(self, x: np.ndarray) -> np.ndarray:
        circuit = self._build_full_circuit(x)
        result = self.sampler.run(circuit, shots=1024).result()
        counts = result.get_counts(circuit)
        probs = np.zeros(2)
        for bitstring, count in counts.items():
            probs[int(bitstring, 2)] = count
        probs /= probs.sum()
        return probs
