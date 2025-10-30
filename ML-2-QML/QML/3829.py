# FraudDetectionHybrid_QML.py
"""Quantum‑centric counterpart of the fraud detection hybrid model.

The module builds a parameterised Qiskit circuit that mirrors the photonic layer
definition and then embeds it in a Qiskit Machine Learning EstimatorQNN.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as _EstimatorQNN

# --------------------------------------------------------------------------- #
# 1.  Parameter container
# --------------------------------------------------------------------------- #
class FraudLayerParameters:
    """Parameters describing a single photonic layer, reused for the quantum circuit."""
    def __init__(
        self,
        bs_theta: float,
        bs_phi: float,
        phases: Tuple[float, float],
        squeeze_r: Tuple[float, float],
        squeeze_phi: Tuple[float, float],
        displacement_r: Tuple[float, float],
        displacement_phi: Tuple[float, float],
        kerr: Tuple[float, float],
    ) -> None:
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to ``[-bound, bound]``."""
    return max(-bound, min(bound, value))


# --------------------------------------------------------------------------- #
# 2.  Quantum circuit construction
# --------------------------------------------------------------------------- #
def _apply_qc_layer(qc: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    """Add one photonic‑inspired layer to a Qiskit circuit."""
    # Beam‑splitter analogy: two single‑qubit rotations
    theta = params.bs_theta if not clip else _clip(params.bs_theta, 5.0)
    phi   = params.bs_phi   if not clip else _clip(params.bs_phi, 5.0)
    qc.rx(theta, 0)
    qc.rz(phi,   1)

    # Phase shifters on each qubit
    for i, phase in enumerate(params.phases):
        qc.ry(phase, i)

    # Squeezing → RZ rotation (clipped)
    for i, (r, _) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qc.rz(_clip(r, 5.0), i)

    # Displacement → RX rotation (clipped)
    for i, (r, _) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qc.rx(_clip(r, 5.0), i)

    # Kerr → RZ rotation (clipped)
    for i, k in enumerate(params.kerr):
        qc.rz(_clip(k, 1.0), i)


def build_fraud_detection_qc(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Create a 2‑qubit Qiskit circuit that mimics the layered photonic structure."""
    qc = QuantumCircuit(2)
    _apply_qc_layer(qc, input_params, clip=False)
    for layer in layers:
        _apply_qc_layer(qc, layer, clip=True)
    return qc


# --------------------------------------------------------------------------- #
# 3.  EstimatorQNN wrapper
# --------------------------------------------------------------------------- #
def create_estimator_qnn(
    input_params: FraudLayerParameters,
    weight_params: FraudLayerParameters,
) -> _EstimatorQNN:
    """
    Build a Qiskit EstimatorQNN that evaluates the circuit prepared by
    ``build_fraud_detection_qc``.  The input parameters are the data
    (e.g. transaction features) while the weight parameters are trained
    during the quantum‑classical hybrid training loop.
    """
    # Build the circuit
    qc = build_fraud_detection_qc(input_params, [])

    # Define Pauli observable (identity on both qubits)
    observable = SparsePauliOp.from_list([("I" * qc.num_qubits, 1)])

    # Parameter placeholders for the estimator
    input_param = Parameter("input")
    weight_param = Parameter("weight")

    # Attach parameters to the circuit
    qc.add_parameters([input_param, weight_param])

    # Instantiate the estimator
    estimator = StatevectorEstimator()

    # Construct EstimatorQNN
    estimator_qnn = _EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[input_param],
        weight_params=[weight_param],
        estimator=estimator,
    )
    return estimator_qnn


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_qc",
    "create_estimator_qnn",
]
