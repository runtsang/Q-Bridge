"""Quantum implementation of the hybrid estimator using Qiskit.

The circuit is a two‑qubit variational network with parameterised rotations
and a CX entangling gate.  An observable Y is measured on qubit 0.
The circuit is wrapped in Qiskit Machine Learning's EstimatorQNN class,
allowing gradient‑based optimisation of both the input and weight parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator


@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic‑style layer (kept for API compatibility)."""
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


def _apply_layer(qc: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    """Apply a single photonic‑style layer using Qiskit primitives."""
    theta = params.bs_theta if not clip else _clip(params.bs_theta, 5)
    phi = params.bs_phi if not clip else _clip(params.bs_phi, 5)

    # Parameterised rotations on each qubit
    qc.ry(theta, 0)
    qc.rx(phi, 0)
    qc.ry(theta, 1)
    qc.rx(phi, 1)

    # Entanglement
    qc.cx(0, 1)

    # Additional single‑qubit rotations
    for i, phase in enumerate(params.phases):
        qc.rz(phase, i)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Construct a full variational circuit mirroring the classical architecture."""
    qc = QuantumCircuit(2)
    _apply_layer(qc, input_params, clip=False)
    for layer in layers:
        _apply_layer(qc, layer, clip=True)
    return qc


def EstimatorQNN(
    input_params: FraudLayerParameters,
    hidden_layers: Iterable[FraudLayerParameters],
) -> QiskitEstimatorQNN:
    """Return a Qiskit EstimatorQNN object ready for training."""
    # Create a circuit with explicit parameter placeholders
    input0 = Parameter("input0")
    input1 = Parameter("input1")
    weight0 = Parameter("weight0")
    weight1 = Parameter("weight1")

    qc = QuantumCircuit(2)
    # Input layer
    qc.ry(input0, 0)
    qc.rx(weight0, 0)
    qc.ry(input1, 1)
    qc.rx(weight1, 1)
    # Entangle
    qc.cx(0, 1)

    # Optional hidden layers (for illustration we reuse the same parameters)
    for _ in hidden_layers:
        qc.ry(input0, 0)
        qc.rx(weight0, 0)
        qc.ry(input1, 1)
        qc.rx(weight1, 1)
        qc.cx(0, 1)

    # Observable: Y on qubit 0
    observable = SparsePauliOp.from_list([("Y0", 1)])

    estimator = Estimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[input0, input1],
        weight_params=[weight0, weight1],
        estimator=estimator,
    )
    return estimator_qnn


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "EstimatorQNN"]
