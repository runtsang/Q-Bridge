from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def fraud_layer_circuit(params: FraudLayerParameters) -> QuantumCircuit:
    qr = QuantumRegister(2, 'q')
    qc = QuantumCircuit(qr)
    qc.ry(params.bs_theta, qr[0])
    qc.ry(params.bs_phi, qr[1])
    qc.rz(params.phases[0], qr[0])
    qc.rz(params.phases[1], qr[1])
    qc.rzz(_clip(params.squeeze_r[0] + params.squeeze_phi[0], 5.0), qr[0], qr[1])
    qc.ry(_clip(params.displacement_r[0], 5.0), qr[0])
    qc.ry(_clip(params.displacement_r[1], 5.0), qr[1])
    qc.rz(_clip(params.kerr[0], 1.0), qr[0])
    qc.rz(_clip(params.kerr[1], 1.0), qr[1])
    return qc

def build_fraud_detection_qc(params: FraudLayerParameters, n_layers: int = 2) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    for _ in range(n_layers):
        layer = fraud_layer_circuit(params)
        qc.compose(layer, inplace=True)
    return qc

def FraudNetHybridQiskit(params: FraudLayerParameters, n_layers: int = 2) -> SamplerQNN:
    qc = build_fraud_detection_qc(params, n_layers)
    sampler = Sampler()
    inputs = ParameterVector('x', 2)
    weight_params = qc.parameters
    return SamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weight_params,
        sampler=sampler,
        interpret=lambda x: x,
    )
