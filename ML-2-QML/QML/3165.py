"""Hybrid quantum model combining QCNN ansatz and photonic fraud detection."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable, Sequence

# ----------------------------------------------------
# Photonic fraud‑detection program (continuous‑variable)
# ----------------------------------------------------
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

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

# ----------------------------------------------------
# QCNN ansatz (discrete‑variable)
# ----------------------------------------------------
def _conv_block(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="ConvLayer")
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = _conv_block(params[idx : idx + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc

def _pool_block(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="PoolLayer")
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = _pool_block(params[idx : idx + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc

def _build_qcnn_ansatz(num_qubits: int = 8) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    fm = ZFeatureMap(num_qubits)
    qc.compose(fm, range(num_qubits), inplace=True)
    qc.compose(_conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    qc.compose(_pool_layer(num_qubits, "p1"), range(num_qubits), inplace=True)
    qc.compose(_conv_layer(num_qubits // 2, "c2"), range(num_qubits // 2, num_qubits), inplace=True)
    qc.compose(_pool_layer(num_qubits // 2, "p2"), range(num_qubits // 2, num_qubits), inplace=True)
    qc.compose(_conv_layer(num_qubits // 4, "c3"), range(3 * num_qubits // 4, num_qubits), inplace=True)
    qc.compose(_pool_layer(num_qubits // 4, "p3"), range(3 * num_qubits // 4, num_qubits), inplace=True)
    return qc

# ----------------------------------------------------
# Hybrid model (quantum side)
# ----------------------------------------------------
class HybridQCNNFraudModel:
    """Quantum hybrid model: QCNN ansatz + photonic fraud detection sub‑circuit."""
    def __init__(
        self,
        qcnn_qubits: int = 8,
        sf_modes: int = 2,
        qc_weight: float = 0.5,
        sf_weight: float = 0.5,
    ) -> None:
        self.qcnn_qubits = qcnn_qubits
        self.sf_modes = sf_modes
        self.qc_weight = qc_weight
        self.sf_weight = sf_weight

        # QCNN ansatz and EstimatorQNN
        self.qcnn_circuit = _build_qcnn_ansatz(qcnn_qubits)
        obs = SparsePauliOp.from_list([("Z" + "I" * (qcnn_qubits - 1), 1)])
        self.qcnn_qnn = EstimatorQNN(
            circuit=self.qcnn_circuit.decompose(),
            observables=obs,
            input_params=self.qcnn_circuit.parameters,
            weight_params=[],
            estimator=Estimator(),
        )

        # Placeholder photonic program; user can supply parameters via set_fraud_params
        dummy_in = FraudLayerParameters(
            bs_theta=0.0,
            bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        dummy_layers = []
        self.sf_program = build_fraud_detection_program(dummy_in, dummy_layers)
        self.sf_engine = sf.Engine("gaussian")

    def set_fraud_params(
        self,
        input_params: FraudLayerParameters,
        hidden_params: Iterable[FraudLayerParameters],
    ) -> None:
        """Replace the photonic sub‑circuit with user‑supplied parameters."""
        self.sf_program = build_fraud_detection_program(input_params, hidden_params)

    def _evaluate_qcnn(self, data: np.ndarray) -> np.ndarray:
        """Run the QCNN ansatz on the classical data."""
        result = self.qcnn_qnn.predict(data)
        return np.asarray(result).reshape(-1)

    def _evaluate_sf(self, data: np.ndarray) -> np.ndarray:
        """Run the photonic fraud detection on the classical data."""
        prog = self.sf_program
        with prog.context as q:
            for i, val in enumerate(data.reshape(-1, 2)):
                Dgate(val[0], 0.0) | q[0]
                Dgate(val[1], 0.0) | q[1]
        state = self.sf_engine.run(prog).state
        exp_z = state.expectation_value('Z', 0)
        return np.asarray(exp_z).reshape(-1)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return weighted combination of QCNN and photonic outputs."""
        qc_out = self._evaluate_qcnn(data)
        sf_out = self._evaluate_sf(data)
        return self.qc_weight * qc_out + self.sf_weight * sf_out

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "HybridQCNNFraudModel",
]
