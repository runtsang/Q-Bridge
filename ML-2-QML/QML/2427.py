"""Quantum fraud detection model using Strawberry Fields and Qiskit EstimatorQNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

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
    clip: bool = True

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not params.clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not params.clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not params.clip else _clip(k, 1)) | modes[i]

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params)
        for layer in layers:
            _apply_layer(q, layer)
    return program

class FraudDetectionHybrid:
    """Quantum fraud detection model leveraging Strawberry Fields and Qiskit EstimatorQNN."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        observable: SparsePauliOp = SparsePauliOp.from_list([("Y"*2, 1)]),
    ) -> None:
        self.program = build_fraud_detection_program(input_params, layers)
        self.observables = [observable]
        self.circuit = self._to_qiskit_circuit()
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=[self.circuit.parameters[0]],
            weight_params=[self.circuit.parameters[1]],
            estimator=self.estimator,
        )

    def _to_qiskit_circuit(self) -> QuantumCircuit:
        # Placeholder mapping from SF to Qiskit; a full implementation would translate gates.
        qc = QuantumCircuit(1)
        qc.h(0)
        return qc

    def forward(self, inputs: Sequence[float]) -> float:
        """Evaluate the quantum circuit expectation for given inputs."""
        params = {self.circuit.parameters[0]: inputs[0], self.circuit.parameters[1]: inputs[1]}
        result = self.qnn.predict(params)
        return float(result)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
