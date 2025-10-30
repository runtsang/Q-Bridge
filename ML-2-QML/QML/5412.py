from __future__ import annotations

from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.circuit.library import ZFeatureMap

class HybridQCNN:
    """Hybrid quantum QCNN circuit combining QCNN, QuantumClassifierModel, and SamplerQNN."""
    def __init__(self, num_qubits: int = 8, depth: int = 3, use_sampler: bool = False) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_sampler = use_sampler
        self.estimator = StatevectorEstimator()
        self.sampler = StatevectorSampler()
        self.circuit, self.input_params, self.weight_params = self._build_circuit()
        if self.use_sampler:
            self.model = SamplerQNN(
                circuit=self.circuit,
                input_params=self.input_params,
                weight_params=self.weight_params,
                sampler=self.sampler,
            )
        else:
            observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])
            self.model = EstimatorQNN(
                circuit=self.circuit,
                observables=observable,
                input_params=self.input_params,
                weight_params=self.weight_params,
                estimator=self.estimator,
            )

    def _single_conv_block(self, params: Iterable[np.ndarray]) -> QuantumCircuit:
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

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        param_vec = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            block_params = param_vec[i * 3 : (i + 2) * 3]
            block = self._single_conv_block(block_params)
            qc.compose(block, [i, i + 1], inplace=True)
            qc.barrier()
        return qc

    def _single_pool_block(self, params: Iterable[np.ndarray]) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_vec = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        for i in range(0, num_qubits, 2):
            block_params = param_vec[i * 3 : (i + 2) * 3]
            block = self._single_pool_block(block_params)
            qc.compose(block, [i, i + 1], inplace=True)
            qc.barrier()
        return qc

    def _variational_ansatz(self, num_qubits: int, depth: int) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        x_params = ParameterVector("x", num_qubits)
        for i in range(num_qubits):
            qc.rx(x_params[i], i)
        for d in range(depth):
            ry_params = ParameterVector(f"ry_{d}", num_qubits)
            for i in range(num_qubits):
                qc.ry(ry_params[i], i)
            for i in range(num_qubits - 1):
                qc.cz(i, i + 1)
        return qc

    def _build_circuit(self) -> tuple[QuantumCircuit, Iterable, Iterable]:
        feature_map = ZFeatureMap(self.num_qubits)
        ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")
        for l in range(self.depth):
            ansatz.compose(self._conv_layer(self.num_qubits, f"c{l}"), inplace=True)
            ansatz.compose(self._pool_layer(self.num_qubits, f"p{l}"), inplace=True)
        ansatz.compose(self._variational_ansatz(self.num_qubits, self.depth), inplace=True)
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        input_params = feature_map.parameters
        weight_params = ansatz.parameters
        return circuit, input_params, weight_params

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        if self.use_sampler:
            return self.model.sample(inputs)
        else:
            return self.model.predict(inputs)
