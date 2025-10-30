from __future__ import annotations

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

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

# --- QCNN helper circuits -----------------------------------------

def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

# --- Hybrid quantum model -----------------------------------------

class FraudDetectionHybridModel:
    """
    Quantum hybrid fraud detection model that stitches a photonic feature extractor
    with a QCNN ansatz into a single EstimatorQNN.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        seed: int | None = None,
    ) -> None:
        self.input_params = input_params
        self.layers = layers
        self.estimator = Estimator()
        self.circuit = self._build_circuit(seed=seed)
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * 7, 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_circuit(self, seed: int | None = None):
        # Photonic feature extractor
        photonic_prog = build_fraud_detection_program(self.input_params, self.layers)
        # QCNN ansatz
        feature_map = ZFeatureMap(8)
        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
        # Assemble full circuit
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)
        return circuit

    @property
    def feature_map(self):
        return ZFeatureMap(8)

    @property
    def ansatz(self):
        # Rebuild ansatz to expose parameters
        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
        return ansatz

    def evaluate(self, data):
        """Evaluate the model on a batch of classical data."""
        return self.qnn.evaluate(data)

    def predict(self, data):
        """Return binary predictions."""
        probs = self.evaluate(data)
        return (probs >= 0.5).astype(int)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybridModel",
]
