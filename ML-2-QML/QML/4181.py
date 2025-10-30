"""Hybrid quantum kernel model with estimator and sampler."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit.circuit import Parameter, ParameterVector
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorEstimator as StatevectorEstimator
from qiskit.primitives import StatevectorSampler as StatevectorSampler


class KernalAnsatz(tq.QuantumModule):
    """Quantum data‑encoding ansatz using a list of gate specs."""

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
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
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


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two batches of samples."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


def EstimatorQNN():
    """Quantum estimator based on a single‑qubit circuit."""
    params1 = [Parameter("input1"), Parameter("weight1")]
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    qc1.ry(params1[0], 0)
    qc1.rx(params1[1], 0)

    observable1 = SparsePauliOp.from_list([("Y" * qc1.num_qubits, 1)])

    estimator = StatevectorEstimator()
    return QiskitEstimatorQNN(
        circuit=qc1,
        observables=observable1,
        input_params=[params1[0]],
        weight_params=[params1[1]],
        estimator=estimator,
    )


def SamplerQNN():
    """Quantum sampler based on a 2‑qubit circuit."""
    inputs2 = ParameterVector("input", 2)
    weights2 = ParameterVector("weight", 4)

    qc2 = QuantumCircuit(2)
    qc2.ry(inputs2[0], 0)
    qc2.ry(inputs2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[0], 0)
    qc2.ry(weights2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[2], 0)
    qc2.ry(weights2[3], 1)

    sampler = StatevectorSampler()
    return QiskitSamplerQNN(
        circuit=qc2,
        input_params=inputs2,
        weight_params=weights2,
        sampler=sampler,
    )


class HybridKernelModel:
    """Unified interface for quantum kernel, estimator, and sampler."""

    def __init__(self) -> None:
        self.kernel = Kernel()
        self.estimator = EstimatorQNN()
        self.sampler = SamplerQNN()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # The QiskitEstimatorQNN returns a Tensor of expectation values
        return self.estimator(x)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        # The QiskitSamplerQNN returns a probability distribution
        return self.sampler(x)


__all__ = ["HybridKernelModel"]
