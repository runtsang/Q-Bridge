"""Hybrid kernel and estimator with quantum components."""

from __future__ import annotations

from typing import Sequence, List
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator


class KernalAnsatz(tq.QuantumModule):
    """Quantum data‑encoding ansatz using a programmable gate list."""
    def __init__(self, func_list: List[dict]) -> None:
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
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Default ansatz: single‑qubit rotations on each wire
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


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b`` using the quantum kernel."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


def EstimatorQNN() -> QiskitEstimatorQNN:
    """Return a Qiskit EstimatorQNN mirroring the classical estimator."""
    # Define a single‑qubit circuit with one data and one weight parameter
    params = [Parameter("input1"), Parameter("weight1")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)
    # Observable: Y on the single qubit
    observable = SparsePauliOp.from_list([("Y", 1)])
    # State‑vector estimator
    estimator = StatevectorEstimator()
    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[params[0]],
        weight_params=[params[1]],
        estimator= estimator
    )


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "EstimatorQNN"]
