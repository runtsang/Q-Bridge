"""Hybrid quantum kernel, estimator, sampler and classifier."""
from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

__all__ = [
    "HybridQuantumModel",
    "QuantumRBFKernel",
    "QuantumEstimator",
    "QuantumSampler",
    "QuantumClassifier",
]


class QuantumRBFKernel(tq.QuantumModule):
    """Programmable RBF kernel implemented with a list of TorchQuantum gates."""
    def __init__(self, func_list: list[dict]) -> None:
        super().__init__()
        self.func_list = func_list
        self.q_device = tq.QuantumDevice(n_wires=len(func_list))

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        return torch.abs(self.q_device.states.view(-1)[0])


class QuantumEstimator:
    """Variational quantum circuit that mirrors the classical EstimatorNN."""
    def __init__(self) -> None:
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])
        estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[params[0]],
            weight_params=[params[1]],
            estimator=estimator,
        )

    def evaluate(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return self.qnn.evaluate(X, weights)


class QuantumSampler:
    """Quantum sampler that outputs a probability distribution over two states."""
    def __init__(self) -> None:
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
        self.qnn = SamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
        )

    def sample(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return self.qnn.sample(X, weights)


class QuantumClassifier:
    """Layered ansatz with explicit encoding and variational parameters."""
    def __init__(self, num_qubits: int = 2, depth: int = 2) -> None:
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)
        circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        self.circuit = circuit
        self.encoding = encoding
        self.weights = weights
        self.observables = observables


class HybridQuantumModel:
    """Facade that groups quantum kernel, estimator, sampler and classifier."""
    def __init__(self, depth: int = 2) -> None:
        self.kernel = QuantumRBFKernel(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.estimator = QuantumEstimator()
        self.sampler = QuantumSampler()
        self.classifier = QuantumClassifier(depth=depth)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

    def evaluate_estimator(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return self.estimator.evaluate(X, weights)

    def sample(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return self.sampler.sample(X, weights)

    def classify(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        # In practice one would bind ``params`` to the circuit and execute.
        # Here we return a dummy array to keep the API complete.
        return np.zeros(len(X), dtype=int)
