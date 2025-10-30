"""Hybrid quantum model combining QCNN ansatz and a 2‑qubit sampler circuit."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 12345


def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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


def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = conv_circuit(params[idx : idx + 3])
        qc.compose(sub, [q1, q2], inplace=True)
        qc.barrier()
        idx += 3
    return qc


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources, sinks, prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=len(sources) * 3)
    idx = 0
    for src, snk in zip(sources, sinks):
        sub = pool_circuit(params[idx : idx + 3])
        qc.compose(sub, [src, snk], inplace=True)
        qc.barrier()
        idx += 3
    return qc


def sampler_circuit() -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    return qc, inputs, weights


class QCNNSamplerQNN:
    """Hybrid quantum neural network: classification via QCNN ansatz and sampling via a 2‑qubit circuit."""

    def __init__(self) -> None:
        # Estimator for expectation‑value classification
        estimator = StatevectorEstimator()
        feature_map = ZFeatureMap(8)

        # Build QCNN ansatz
        ansatz = QuantumCircuit(8, name="QCNN_Ansatz")
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.classifier = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

        # Sampler circuit
        sampler_qc, inp, wgt = sampler_circuit()
        self.sampler = SamplerQNN(
            circuit=sampler_qc,
            input_params=inp,
            weight_params=wgt,
            sampler=StatevectorSampler(),
        )

    def forward(self, inputs: np.ndarray) -> tuple[float, np.ndarray]:
        """Return classification expectation and sampled distribution."""
        class_out = self.classifier.predict(inputs)
        samp_out = self.sampler.sample(inputs)
        return class_out, samp_out


def QCNNSamplerQNN() -> QCNNSamplerQNN:
    """Factory returning the hybrid quantum neural network."""
    return QCNNSamplerQNN()


__all__ = ["QCNNSamplerQNN", "QCNNSamplerQNN"]
