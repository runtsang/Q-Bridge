"""Hybrid quantum‑classical classifier that combines a Qiskit variational circuit with a Strawberry Fields fraud detection program.

Features:
* build_classifier_circuit – a layered ansatz with explicit encoding and observables.
* build_fraud_detection_program – a photonic program mirroring the classical fraud‑detection structure.
* EstimatorQNN – a Qiskit EstimatorQNN that can be used as a hybrid backend.
* HybridClassifier – a simple container exposing the circuit and its metadata for immediate use.

The module is fully importable and can be plugged into Qiskit or Pennylane workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator

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


def _apply_layer(modes: Iterable, params: FraudLayerParameters, *, clip: bool) -> None:
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


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp.from_list([("I" * i + "Z" + "I" * (num_qubits - i - 1), 1)])
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


def EstimatorQNN() -> QiskitEstimatorQNN:
    """Return a Qiskit EstimatorQNN that wraps a tiny variational circuit."""
    param_input = Parameter("input1")
    param_weight = Parameter("weight1")
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    qc1.ry(param_input, 0)
    qc1.rx(param_weight, 0)

    observable1 = SparsePauliOp.from_list([("Y", 1)])

    estimator = Estimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc1,
        observables=observable1,
        input_params=[param_input],
        weight_params=[param_weight],
        estimator=estimator,
    )
    return estimator_qnn


class HybridClassifier:
    """Container exposing the quantum circuit and its metadata for immediate use."""
    def __init__(self, num_qubits: int, depth: int) -> None:
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "build_classifier_circuit",
    "EstimatorQNN",
    "HybridClassifier",
]
