"""Hybrid quantum model integrating TorchQuantum kernel, Qiskit QCNN, and SamplerQNN."""
from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN as QiskitSamplerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms import QuantumKernel
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.providers.aer import AerSimulator

class QuantumKernelAnsatz(tq.QuantumModule):
    """Programmable ansatz that encodes classical data into a 4‑qubit device."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernelModule(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernelAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class QCNNQuantum:
    """Full QCNN construction using Qiskit and a classical estimator."""
    def __init__(self):
        algorithm_globals.random_seed = 12345
        self.estimator = StatevectorEstimator()
        self._build_circuit()

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(self._conv_circuit(params[param_index:param_index+3]), qubits=[q1, q2], inplace=True)
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.compose(self._conv_circuit(params[param_index:param_index+3]), qubits=[q1, q2], inplace=True)
            param_index += 3
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources, sinks, param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, snk in zip(sources, sinks):
            qc.compose(self._pool_circuit(params[param_index:param_index+3]), qubits=[src, snk], inplace=True)
            param_index += 3
        return qc

    def _build_circuit(self):
        feature_map = ZFeatureMap(8)
        ansatz = QuantumCircuit(8)
        ansatz.compose(self._conv_layer(8, "c1"), inplace=True)
        ansatz.compose(self._pool_layer([0,1,2,3],[4,5,6,7],"p1"), inplace=True)
        ansatz.compose(self._conv_layer(4, "c2"), inplace=True)
        ansatz.compose(self._pool_layer([0,1],[2,3],"p2"), inplace=True)
        ansatz.compose(self._conv_layer(2, "c3"), inplace=True)
        ansatz.compose(self._pool_layer([0],[1],"p3"), inplace=True)
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator
        )

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return QCNN probabilities for each sample."""
        return np.array([self.qnn.predict(qc_input=x.reshape(1, -1))[0] for x in data])

class SamplerQuantum:
    """Quantum SamplerQNN using Qiskit StatevectorSampler."""
    def __init__(self):
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
        sampler = StatevectorSampler()
        self.sampler_qnn = QiskitSamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler
        )

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return sampled probability distributions for each input."""
        return np.array([self.sampler_qnn.predict(qc_input=x.reshape(1, -1))[0] for x in data])

class HybridKernelQCNNQML:
    """Hybrid quantum model combining TorchQuantum kernel, QCNN, and SamplerQNN."""
    def __init__(self) -> None:
        self.kernel = QuantumKernelModule()
        self.qcnn = QCNNQuantum()
        self.sampler = SamplerQuantum()

    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute quantum kernel Gram matrix using TorchQuantum."""
        a_t = torch.tensor(a, dtype=torch.float32)
        b_t = torch.tensor(b, dtype=torch.float32)
        mat = np.array([[self.kernel(x, y).item() for y in b_t] for x in a_t])
        return mat

    def qcnn_predict(self, data: np.ndarray) -> np.ndarray:
        return self.qcnn.predict(data)

    def sampler_predict(self, data: np.ndarray) -> np.ndarray:
        return self.sampler.predict(data)

__all__ = ["HybridKernelQCNNQML", "QuantumKernelModule", "QCNNQuantum", "SamplerQuantum"]
