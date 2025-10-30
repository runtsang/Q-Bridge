"""Hybrid quantum sampler QNN combining SamplerQNN and FraudDetection concepts."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class _FraudLayerQuantum:
    def __init__(self, params: dict, clip: bool = False) -> None:
        self.params = params
        self.clip = clip

    def apply(self, qc: QuantumCircuit) -> None:
        # Input rotations
        qc.ry(self.params["input_0"], 0)
        qc.ry(self.params["input_1"], 1)
        # Entanglement
        qc.cx(0, 1)
        # Parameterized rotations
        qc.ry(self.params["weight_0"], 0)
        qc.ry(self.params["weight_1"], 1)
        qc.cx(0, 1)
        qc.ry(self.params["weight_2"], 0)
        qc.ry(self.params["weight_3"], 1)
        # Fraud-like layer
        for i in range(2):
            phase = self.params["phase_{}".format(i)]
            qc.rz(phase, i)
        for i in range(2):
            r = self.params["squeeze_r"][i]
            phi = self.params["squeeze_phi"][i]
            if self.clip:
                r = _clip(r, 5.0)
            # Placeholder for Sgate: use s gate as a simple representation
            qc.s(i)
        # Additional gates can be added if desired

class HybridSamplerQNN(QSamplerQNN):
    def __init__(self, circuit: QuantumCircuit, input_params, weight_params, sampler, clip: bool = False) -> None:
        super().__init__(circuit=circuit, input_params=input_params, weight_params=weight_params, sampler=sampler)
        self.clip = clip

    @classmethod
    def from_params(cls, input_params: dict, layers: list[dict]) -> "HybridSamplerQNN":
        qc = QuantumCircuit(2)
        # First layer (no clipping)
        first_layer = _FraudLayerQuantum(input_params, clip=False)
        first_layer.apply(qc)
        # Subsequent layers (with clipping)
        for layer_params in layers:
            layer = _FraudLayerQuantum(layer_params, clip=True)
            layer.apply(qc)
        sampler = StatevectorSampler()
        return cls(circuit=qc,
                   input_params=ParameterVector("input", 2),
                   weight_params=ParameterVector("weight", 4),
                   sampler=sampler)

def SamplerQNN() -> HybridSamplerQNN:
    # Example placeholder parameters
    input_params = {
        "input_0": ParameterVector("input", 2)[0],
        "input_1": ParameterVector("input", 2)[1],
        "weight_0": ParameterVector("weight", 4)[0],
        "weight_1": ParameterVector("weight", 4)[1],
        "weight_2": ParameterVector("weight", 4)[2],
        "weight_3": ParameterVector("weight", 4)[3],
        "phase_0": ParameterVector("phase0", 1)[0],
        "phase_1": ParameterVector("phase1", 1)[0],
        "squeeze_r": (ParameterVector("squeeze_r0", 1)[0], ParameterVector("squeeze_r1", 1)[0]),
        "squeeze_phi": (ParameterVector("squeeze_phi0", 1)[0], ParameterVector("squeeze_phi1", 1)[0]),
    }
    layers = [input_params]  # placeholder; real layers should be provided
    return HybridSamplerQNN.from_params(input_params, layers)

__all__ = ["HybridSamplerQNN", "SamplerQNN"]
