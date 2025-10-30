"""Quantum‑centric hybrid kernel and estimator.

This module defines :class:`HybridKernelMethod` that implements:
* a TorchQuantum variational kernel,
* a Qiskit-based expectation‑value evaluator for parametric circuits,
* a combined interface that mirrors the classical module.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class HybridKernelMethod:
    """Quantum‑centric hybrid kernel and estimator.

    The class combines a TorchQuantum variational ansatz with a Qiskit
    expectation‑value evaluator.  It provides the same public API as the
    classical counterpart so that downstream code can switch back‑ends
    without modification.
    """

    def __init__(
        self,
        n_wires: int = 4,
        func_list: List[dict] | None = None,
    ) -> None:
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.func_list = func_list or [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]
        self.ansatz = self._build_ansatz(self.func_list)
        self.circuit = self._build_circuit()

    def _build_ansatz(self, func_list: List[dict]) -> tq.QuantumModule:
        class KernalAnsatz(tq.QuantumModule):
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
        return KernalAnsatz(func_list)

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_wires)
        for i in range(self.n_wires):
            theta = Parameter(f"theta_{i}")
            qc.ry(theta, i)
            qc.cx(i, (i + 1) % self.n_wires)
        return qc

    # Placeholder for compatibility; not used in quantum branch
    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-1.0 * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        return np.array([[self._quantum_kernel(x, y).item() for y in b] for x in a])

    def _quantum_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_qc = self._bind_circuit(values)
            state = Statevector.from_instruction(bound_qc)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _bind_circuit(self, params: Sequence[float]) -> QuantumCircuit:
        if len(params)!= self.n_wires:
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = {f"theta_{i}": p for i, p in enumerate(params)}
        bound = self.circuit.assign_parameters(mapping, inplace=False)
        return bound


__all__ = ["HybridKernelMethod"]
