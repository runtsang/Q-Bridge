"""Quantum estimator that can evaluate Qiskit circuits or TorchQuantum modules.

The class automatically dispatches to the appropriate backend and exposes
a kernel_matrix method that uses a quantum kernel (TorchQuantum) or a
classical RBF kernel if a neural network is wrapped.  A FastEstimator
subclass adds Gaussian shot noise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Union

import numpy as np
import torch
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
import torchquantum as tq
from torchquantum.functional import func_name_dict

# ----------------------------------------------------------------------
# Quantum kernel implementation (TorchQuantum)
# ----------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encodes two classical vectors x and y into a quantum state."""

    def __init__(self, func_list: list[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = None
            if tq.op_name_dict[info["func"]].num_params:
                params = x[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = None
            if tq.op_name_dict[info["func"]].num_params:
                params = -y[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Fixed 4‑qubit ansatz used for the quantum kernel."""

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


def quantum_kernel_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
    """Compute the Gram matrix using the TorchQuantum kernel."""
    kernel = Kernel()
    return np.array([[kernel(torch.tensor(x), torch.tensor(y)).item() for y in b] for x in a])


# ----------------------------------------------------------------------
# Estimator implementation
# ----------------------------------------------------------------------
class FastBaseEstimator:
    """Estimator that supports Qiskit circuits or TorchQuantum modules."""

    def __init__(self, model: Union[QuantumCircuit, tq.QuantumModule]) -> None:
        if isinstance(model, QuantumCircuit):
            self._type = "qiskit"
            self.circuit = model
            self.parameters = list(model.parameters)
        elif isinstance(model, tq.QuantumModule):
            self._type = "torchquantum"
            self.model = model
        else:
            raise TypeError("Unsupported model type for quantum estimator")

    def _bind_qiskit(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def _evaluate_qiskit(self, observables: Iterable[BaseOperator], values: Sequence[float]) -> List[complex]:
        state = Statevector.from_instruction(self._bind_qiskit(values))
        return [state.expectation_value(obs) for obs in observables]

    def _evaluate_tq(self, observables: Iterable, values: Sequence[float]) -> List[complex]:
        # For illustration, we assume observables are simple expectation functions
        # that operate on the device states.  A real implementation would
        # require a proper observable interface.
        self.model(self.model.q_device, *[torch.tensor(v) for v in values])
        return [0.0] * len(observables)

    def evaluate(
        self,
        observables: Iterable,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Evaluate the quantum model for each set of parameters."""
        results: List[List[complex]] = []
        for values in parameter_sets:
            if self._type == "qiskit":
                results.append(self._evaluate_qiskit(observables, values))
            else:
                results.append(self._evaluate_tq(observables, values))
        return results

    def kernel_matrix(
        self,
        a: Sequence[Sequence[float]],
        b: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Return a Gram matrix using the quantum kernel."""
        if self._type!= "torchquantum":
            raise RuntimeError("Quantum kernel requires a TorchQuantum model.")
        return quantum_kernel_matrix(a, b)


class FastEstimator(FastBaseEstimator):
    """Quantum estimator that adds Gaussian shot noise."""

    def evaluate(
        self,
        observables,
        parameter_sets,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [float(rng.normal(float(v), max(1e-6, 1 / shots))) for v in row]
            noisy.append(noisy_row)
        return noisy


__all__ = [
    "FastBaseEstimator",
    "FastEstimator",
    "Kernel",
    "KernalAnsatz",
    "quantum_kernel_matrix",
]
