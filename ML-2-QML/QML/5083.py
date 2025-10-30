"""Quantum estimators that mirror the classical hybrid architecture."""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence, Callable, Any

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Sampler as QiskitSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN


# ------------------------------------------------------------
# Base estimator for a parametric Qiskit circuit
# ------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametric Qiskit circuit."""

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


# ------------------------------------------------------------
# TorchQuantum kernel components
# ------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""

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


class TorchQuantumKernelEstimator:
    """Computes a quantum kernel via a fixed TorchQuantum ansatz."""

    def __init__(self, n_wires: int = 4):
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
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

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])


# ------------------------------------------------------------
# SamplerQNN estimator (Qiskit)
# ------------------------------------------------------------
class SamplerQNNEstimator:
    """Wraps a qiskit_machine_learning SamplerQNN circuit."""

    def __init__(self, circuit: QuantumCircuit, input_params: Sequence, weight_params: Sequence):
        self.sampler = QiskitSampler()
        self.qnn = QiskitSamplerQNN(
            circuit=circuit,
            input_params=input_params,
            weight_params=weight_params,
            sampler=self.sampler,
        )

    def sample(self, inputs: Sequence[Sequence[float]]) -> List[np.ndarray]:
        return [self.qnn.sample(inp) for inp in inputs]


# ------------------------------------------------------------
# Combined quantum estimator
# ------------------------------------------------------------
class CombinedQuantumEstimator(FastBaseEstimator):
    """
    Unified quantum estimator that can operate with a Qiskit circuit,
    a TorchQuantum kernel, or a SamplerQNN.
    """

    def __init__(
        self,
        circuit: QuantumCircuit | None = None,
        *,
        tq_kernel: bool = False,
        tq_n_wires: int = 4,
        sampler_qnn: bool = False,
    ):
        if circuit is None and not tq_kernel and not sampler_qnn:
            raise ValueError("Must provide either a circuit or enable a quantum kernel or SamplerQNN.")

        self.circuit = circuit
        self.base = FastBaseEstimator(circuit) if circuit is not None else None
        self.tq_kernel = TorchQuantumKernelEstimator(n_wires=tq_n_wires) if tq_kernel else None
        self.sampler_qnn = None
        if sampler_qnn:
            if circuit is None:
                raise ValueError("SamplerQNN requires a circuit.")
            self.sampler_qnn = SamplerQNNEstimator(circuit, circuit.parameters, circuit.parameters)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        if self.base is not None:
            return self.base.evaluate(observables, parameter_sets)

        if self.tq_kernel is not None:
            # parameter_sets are pairs (x, y)
            return [[self.tq_kernel.forward(torch.tensor(x, dtype=torch.float32),
                                            torch.tensor(y, dtype=torch.float32)).item()] for x, y in parameter_sets]

        if self.sampler_qnn is not None:
            return self.sampler_qnn.sample(parameter_sets)

        raise RuntimeError("Estimator not properly configured.")


__all__ = [
    "FastBaseEstimator",
    "TorchQuantumKernelEstimator",
    "SamplerQNNEstimator",
    "CombinedQuantumEstimator",
]
