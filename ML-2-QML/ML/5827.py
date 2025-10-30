"""Hybrid kernel and estimator implementation for classical and quantum models."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

# --------------------------------------------------------------------------- #
# Classical / Quantum kernel ansatzes
# --------------------------------------------------------------------------- #

class ClassicalKernelAnsatz(nn.Module):
    """RBF kernel implemented as a PyTorch module."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class QuantumKernelAnsatz:
    """Encodes data into a fixed quantum circuit using TorchQuantum."""

    def __init__(self, func_list: List[dict]) -> None:
        import torchquantum as tq
        from torchquantum.functional import func_name_dict

        self.tq = tq
        self.func_name_dict = func_name_dict
        self.func_list = func_list

    def forward(self, q_device: "tq.QuantumDevice", x: torch.Tensor, y: torch.Tensor) -> None:
        self.tq.static_support(self._forward_impl)(q_device, x, y)

    def _forward_impl(self, q_device: "tq.QuantumDevice", x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if self.tq.op_name_dict[info["func"]].num_params
                else None
            )
            self.func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if self.tq.op_name_dict[info["func"]].num_params
                else None
            )
            self.func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


# --------------------------------------------------------------------------- #
# Hybrid kernel module
# --------------------------------------------------------------------------- #

class HybridKernel(nn.Module):
    """Kernel that can operate in classical, quantum, or hybrid mode."""

    def __init__(
        self,
        mode: str = "classical",
        *,
        gamma: float = 1.0,
        func_list: List[dict] | None = None,
        n_wires: int = 4,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        if self.mode == "classical":
            self.ansatz = ClassicalKernelAnsatz(gamma)
        elif self.mode == "quantum":
            if func_list is None:
                func_list = [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            self.ansatz = QuantumKernelAnsatz(func_list)
            self.q_device = self.ansatz.tq.QuantumDevice(n_wires=n_wires)
        elif self.mode == "hybrid":
            self.classical = ClassicalKernelAnsatz(gamma)
            if func_list is None:
                func_list = [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            self.quantum = QuantumKernelAnsatz(func_list)
            self.q_device = self.quantum.tq.QuantumDevice(n_wires=n_wires)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "classical":
            return self.ansatz(x, y).squeeze()
        if self.mode == "quantum":
            self.ansatz(self.q_device, x, y)
            return torch.abs(self.q_device.states.view(-1)[0])
        # hybrid: weighted sum of classical and quantum contributions
        class_val = self.classical(x, y).squeeze()
        self.quantum(self.q_device, x, y)
        quantum_val = torch.abs(self.q_device.states.view(-1)[0])
        return 0.7 * class_val + 0.3 * quantum_val

def hybrid_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], mode: str = "classical") -> np.ndarray:
    """Compute Gram matrix using the specified kernel mode."""
    kernel = HybridKernel(mode=mode)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
# Hybrid estimator
# --------------------------------------------------------------------------- #

class HybridFastEstimator:
    """Estimator that can evaluate both classical PyTorch models and Qiskit circuits."""

    def __init__(
        self,
        model: Union[nn.Module, "qiskit.circuit.QuantumCircuit"],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed
        self._is_quantum = hasattr(model, "assign_parameters")  # naive check for Qiskit

    def _ensure_batch(self, values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor] | "qiskit.quantum_info.operators.base_operator.BaseOperator"],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        obs = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        if self._is_quantum:
            from qiskit.circuit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            from qiskit.quantum_info.operators.base_operator import BaseOperator

            circuit = self.model
            parameters = list(circuit.parameters)

            def bind(vals: Sequence[float]) -> QuantumCircuit:
                if len(vals)!= len(parameters):
                    raise ValueError("Parameter count mismatch.")
                mapping = dict(zip(parameters, vals))
                return circuit.assign_parameters(mapping, inplace=False)

            for vals in parameter_sets:
                state = Statevector.from_instruction(bind(vals))
                row = [float(state.expectation_value(obs)) for obs in obs]
                results.append(row)
        else:
            self.model.eval()
            with torch.no_grad():
                for vals in parameter_sets:
                    inputs = self._ensure_batch(vals)
                    outputs = self.model(inputs)
                    row: List[float] = []
                    for observable in obs:
                        val = observable(outputs)
                        if isinstance(val, torch.Tensor):
                            scalar = float(val.mean().cpu())
                        else:
                            scalar = float(val)
                        row.append(scalar)
                    results.append(row)

        # Add shot noise if requested
        if self.shots is not None:
            rng = np.random.default_rng(self.seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = [
    "ClassicalKernelAnsatz",
    "QuantumKernelAnsatz",
    "HybridKernel",
    "hybrid_kernel_matrix",
    "HybridFastEstimator",
]
