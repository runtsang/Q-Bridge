"""Unified FastEstimator for quantum circuits, kernels and samplers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union, Optional

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# ----- Quantum Kernel ----------
class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a list of parameterised gates."""
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
    """Quantum kernel based on a fixed ansatz."""
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
    """Compute Gram matrix via the quantum kernel."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----- Sampler QNN ----------
def SamplerQNN() -> tq.QuantumModule:
    """Return a simple parameterised sampler QNN."""
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

    from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
    from qiskit.primitives import StatevectorSampler as Sampler

    sampler = Sampler()
    return QiskitSamplerQNN(
        circuit=qc2,
        input_params=inputs2,
        weight_params=weights2,
        sampler=sampler,
    )

# ----- Unified Estimator ----------
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class FastGenEstimator:
    """Estimator capable of evaluating quantum circuits, quantum kernels and sampler QNNs."""
    def __init__(self, model: Union[QuantumCircuit, Kernel, tq.QuantumModule]) -> None:
        self.model = model
        if isinstance(model, QuantumCircuit):
            self._mode = "circuit"
        elif isinstance(model, Kernel):
            self._mode = "kernel"
        elif isinstance(model, tq.QuantumModule):
            self._mode = "sampler"
        else:
            raise TypeError("Unsupported model type")

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if not isinstance(self.model, QuantumCircuit):
            raise RuntimeError("bind only for raw circuits")
        if len(param_values)!= len(self.model.parameters):
            raise ValueError("Parameter count mismatch")
        mapping = dict(zip(self.model.parameters, param_values))
        return self.model.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Evaluate observables or sample probabilities for each parameter set."""
        if self._mode == "kernel":
            X = [torch.tensor(p, dtype=torch.float32) for p in parameter_sets]
            Y = X
            gram = kernel_matrix(X, Y)
            return gram.tolist()

        if self._mode == "sampler":
            results: List[List[complex]] = []
            for values in parameter_sets:
                inp_params = self.model.input_params
                wgt_params = self.model.weight_params
                if len(values)!= len(inp_params) + len(wgt_params):
                    raise ValueError("Parameter count mismatch for SamplerQNN.")
                inp_vals = values[: len(inp_params)]
                wgt_vals = values[len(inp_params) :]
                probs = self.model.sample(
                    input_params=inp_vals,
                    weight_params=wgt_vals,
                    shots=shots,
                    seed=seed,
                ).toarray()
                results.append(probs.tolist())
            return results

        # Circuit mode
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def kernel_matrix(
        self,
        data_a: Sequence[Sequence[float]],
        data_b: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Convenience routine for quantum kernel Gram matrix."""
        if self._mode!= "kernel":
            raise RuntimeError("kernel_matrix only available for Kernel models")
        X = [torch.tensor(x, dtype=torch.float32) for x in data_a]
        Y = [torch.tensor(y, dtype=torch.float32) for y in data_b]
        return kernel_matrix(X, Y)

__all__ = [
    "FastGenEstimator",
    "Kernel",
    "KernalAnsatz",
    "kernel_matrix",
    "SamplerQNN",
]
