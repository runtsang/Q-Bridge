from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer for the quantum model."""
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

def _apply_layer(qc: QuantumCircuit, params: FraudLayerParameters, clip: bool) -> None:
    # Beam‑splitter approximation with Ry and Rz
    qc.ry(params.bs_theta, 0)
    qc.rz(params.bs_phi, 0)
    qc.ry(params.bs_theta, 1)
    qc.rz(params.bs_phi, 1)

    # Phases
    for i, phase in enumerate(params.phases):
        qc.rz(phase, i)

    # Squeezing approximation
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qc.rx(_clip(r, 5.0) if clip else r, i)
        qc.rz(_clip(phi, 5.0) if clip else phi, i)

    # Displacement approximation
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qc.rx(_clip(r, 5.0) if clip else r, i)
        qc.rz(_clip(phi, 5.0) if clip else phi, i)

    # Kerr approximation
    for i, k in enumerate(params.kerr):
        qc.rz(_clip(k, 1.0) if clip else k, i)

def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    _apply_layer(qc, input_params, clip=False)
    for layer in layers:
        _apply_layer(qc, layer, clip=True)
    return qc

class EstimatorQNN:
    """Tiny parameterised quantum circuit that returns a scalar expectation."""
    def __init__(self) -> None:
        self.params = [Parameter("input1"), Parameter("weight1")]
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.params[0], 0)
        self.circuit.rx(self.params[1], 0)
        self.backend = Aer.get_backend("statevector_simulator")

    def evaluate(self, input_val: float, weight_val: float) -> float:
        bound = {self.params[0]: input_val, self.params[1]: weight_val}
        bound_qc = self.circuit.bind_parameters(bound)
        result = execute(bound_qc, self.backend).result()
        state = result.get_statevector()
        y_pauli = np.array([[0, -1j], [1j, 0]])
        exp = np.vdot(state, y_pauli @ state).real
        return exp

class FraudEstimator:
    """Hybrid quantum fraud detection model: photonic‑style circuit + EstimatorQNN."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.fraud_circuit = build_fraud_detection_circuit(input_params, self.layers)
        self.estimator_qnn = EstimatorQNN()
        self.backend = Aer.get_backend("statevector_simulator")

    def forward(self, input_val: float, weight_val: float) -> float:
        # Execute photonic circuit
        result = execute(self.fraud_circuit, self.backend).result()
        state = result.get_statevector()
        # Feature: probability of |00>
        feature = np.abs(state[0]) ** 2
        # Pass feature to EstimatorQNN
        return self.estimator_qnn.evaluate(feature, weight_val)

__all__ = ["FraudLayerParameters", "build_fraud_detection_circuit",
           "EstimatorQNN", "FraudEstimator"]
