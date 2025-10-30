"""Hybrid estimator for Qiskit quantum circuits with optional shot noise and a fixed quantum kernel."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class KernalAnsatz(tq.QuantumModule):
    """Encodes two classical data vectors into a quantum state and reverses the encoding."""

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
    """Fixed four‑qubit ansatz used to compute a quantum kernel."""

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


def _quantum_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class HybridBaseEstimator:
    """Evaluate a parametrised Qiskit circuit, optionally add shot noise, and compute a quantum kernel."""

    def __init__(self, circuit: QuantumCircuit, shots: Optional[int] = None, seed: Optional[int] = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        self.seed = seed

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectations for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if self.shots is None:
            return results

        rng = np.random.default_rng(self.seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [complex(rng.normal(row_i.real, max(1e-6, 1 / self.shots)),
                                 rng.normal(row_i.imag, max(1e-6, 1 / self.shots))) for row_i in row]
            noisy.append(noisy_row)
        return noisy

    def kernel_matrix(self, data_x: Sequence[torch.Tensor], data_y: Optional[Sequence[torch.Tensor]] = None) -> np.ndarray:
        """Return the quantum kernel Gram matrix between two datasets."""
        if data_y is None:
            data_y = data_x
        return _quantum_kernel_matrix(data_x, data_y)


__all__ = ["HybridBaseEstimator", "Kernel", "KernalAnsatz"]
