from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator

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

def build_fraud_detection_photonic_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the photonic part."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def create_variational_circuit() -> QuantumCircuit:
    """Variational circuit that mirrors the EstimatorQNN example."""
    theta = Parameter("θ")
    phi = Parameter("φ")
    w = Parameter("w")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.ry(theta, 0)
    qc.rx(w, 0)
    qc.cx(0, 1)
    qc.rz(phi, 1)
    qc.rx(w, 1)
    return qc

class FraudDetectionHybrid:
    """
    Quantum implementation of the fraud‑detection model.
    Combines a photonic program with a variational Qiskit circuit.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.input_params = input_params
        self.layers = layers
        self.photonic_program = build_fraud_detection_photonic_program(input_params, layers)
        self.variational_circuit = create_variational_circuit()
        # Prepare EstimatorQNN instance for expectation evaluation
        self.estimator = Estimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.variational_circuit,
            observables=[SparsePauliOp.from_list([("Z" * self.variational_circuit.num_qubits, 1)])],
            input_params=[self.variational_circuit.parameters[0]],
            weight_params=[self.variational_circuit.parameters[1]],
            estimator=self.estimator,
        )

    def get_photonic_program(self) -> sf.Program:
        """Return the Strawberry Fields photonic program."""
        return self.photonic_program

    def get_variational_circuit(self) -> QuantumCircuit:
        """Return the Qiskit variational circuit."""
        return self.variational_circuit

    def evaluate_variational(self, params: dict) -> float:
        """
        Evaluate the variational circuit with the given parameter mapping.
        Returns the expectation value of the observable.
        """
        return float(self.estimator_qnn.predict(params))

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
