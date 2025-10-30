"""Hybrid kernel and estimator using TorchQuantum for kernel and Qiskit for expectation evaluation."""

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence

class KernelAnsatz(tq.QuantumModule):
    """Encodes data via a list of quantum gates."""
    def __init__(self, gates: Sequence[dict]) -> None:
        super().__init__()
        self.gates = gates

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for g in self.gates:
            params = x[:, g["input_idx"]] if tq.op_name_dict[g["func"]].num_params else None
            func_name_dict[g["func"]](q_device, wires=g["wires"], params=params)
        for g in reversed(self.gates):
            params = -y[:, g["input_idx"]] if tq.op_name_dict[g["func"]].num_params else None
            func_name_dict[g["func"]](q_device, wires=g["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Fixed quantum kernel using a predefined ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernelAnsatz(
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
    """Compute quantum Gram matrix."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class FastBaseEstimator:
    """Expectations of a Qiskit circuit for multiple parameter sets."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self._params, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class HybridKernelEstimator:
    """Combines the quantum kernel with a fast Qiskit estimator."""
    def __init__(self, kernel: Kernel, estimator: FastBaseEstimator) -> None:
        self.kernel = kernel
        self.estimator = estimator

    def gram_matrix(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(X, Y)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        return self.estimator.evaluate(observables, parameter_sets)

__all__ = ["KernelAnsatz", "Kernel", "kernel_matrix", "FastBaseEstimator", "HybridKernelEstimator"]
