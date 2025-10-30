"""Quantum implementation of SharedEstimator combining EstimatorQNN, SamplerQNN, and quantum kernel."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union
import numpy as np
from qiskit.circuit import Parameter, ParameterVector
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import Estimator as StateEstimator, Sampler as StateSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN, SamplerQNN as QSamplerQNN

# Quantum kernel using TorchQuantum
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class KernalAnsatz(tq.QuantumModule):
    """Quantum RBF kernel ansatz."""
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

class Kernel(tq.QuantumModule):
    """Quantum kernel module."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def quantum_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class SharedEstimator:
    """Quantum counterpart of SharedEstimator."""
    def __init__(self, gamma: float = 1.0) -> None:
        # Regression circuit
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 1)
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(self.input_params[0], 0)
        qc.rx(self.weight_params[0], 0)
        self.circuit = qc

        # Observables
        self.observable = SparsePauliOp.from_list([("Y", 1)])

        # Estimator primitive
        self.estimator = StateEstimator()
        self.estimator_qnn = QEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.input_params[0]],
            weight_params=[self.weight_params[0]],
            estimator=self.estimator,
        )

        # Sampler circuit
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc2 = QuantumCircuit(2)
        qc2.ry(inputs[0], 0)
        qc2.ry(inputs[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights[0], 0)
        qc2.ry(weights[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights[2], 0)
        qc2.ry(weights[3], 1)
        self.sampler_circuit = qc2
        self.sampler = StateSampler()
        self.sampler_qnn = QSamplerQNN(
            circuit=qc2,
            input_params=inputs,
            weight_params=weights,
            sampler=self.sampler,
        )

        # Quantum kernel
        self.kernel = Kernel()
        self.gamma = gamma

    def evaluate(
        self,
        observables: Iterable[Union[SparsePauliOp, List[SparsePauliOp]]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Expectation value evaluation using the EstimatorQNN primitive."""
        return self.estimator_qnn.evaluate(observables, parameter_sets)

    def sample(self, parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Sample from the SamplerQNN."""
        return self.sampler_qnn.sample(parameter_sets)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix via quantum kernel."""
        return quantum_kernel_matrix(a, b)

__all__ = ["SharedEstimator", "quantum_kernel_matrix"]
