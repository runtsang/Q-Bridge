"""Quantum‑kernel and fast‑estimator utilities built on TorchQuantum and Qiskit.

The module mirrors the classical counterpart but replaces the
Gaussian ansatz with a parametrised quantum circuit.  It also
provides a lightweight estimator that evaluates expectation values
for a list of observables and optionally injects shot noise.

Key additions compared to the seed files:

* :class:`KernalAnsatz` now accepts a list of gate specifications
  and supports both forward and reverse encoding.
* :class:`Kernel` exposes a ``forward`` that re‑uses the same
  circuit for every pair of inputs, thereby keeping the batch size
  fixed to one.
* :class:`FastQuantumEstimator` extends the simple
  :class:`FastBaseEstimator` from the QML seed with a noise model.
* :class:`KernelRidge` implements kernel ridge regression
  on the quantum kernel, re‑using the same API as the classical
  :class:`KernelRidge`.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

# -- Quantum ansatz ------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Programmable quantum encoding circuit.

    Parameters
    ----------
    func_list : List[Dict]
        Each entry specifies ``input_idx``, ``func`` and ``wires``.
    """
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
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def quantum_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_wires: int = 4) -> np.ndarray:
    """Compute the Gram matrix using the quantum kernel."""
    kernel = Kernel(n_wires=n_wires)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# -- Fast estimator utilities -----------------------------------------------
class FastBaseEstimator:
    """Expectation‑value evaluator for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

class FastQuantumEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to the deterministic expectation values."""
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
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
            noisy_row = [rng.normal(val.real, 1 / shots) + 1j * rng.normal(val.imag, 1 / shots) for val in row]
            noisy.append(noisy_row)
        return noisy

# -- Quantum kernel ridge regression -----------------------------------------
class KernelRidge(nn.Module):
    """Kernel ridge regression on a quantum kernel."""
    def __init__(self, kernel: tq.QuantumModule, alpha: float = 1.0) -> None:
        super().__init__()
        self.kernel = kernel
        self.alpha = alpha
        self.coef_ = None
        self.X_train_ = None

    def fit(self, X: Sequence[torch.Tensor], y: torch.Tensor) -> None:
        K = quantum_kernel_matrix(X, X, n_wires=self.kernel.n_wires)
        K += self.alpha * np.eye(K.shape[0])
        self.coef_ = torch.from_numpy(np.linalg.solve(K, y.numpy()))
        self.X_train_ = X

    def predict(self, X: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.coef_ is None or self.X_train_ is None:
            raise RuntimeError("Model is not fitted.")
        K_test = quantum_kernel_matrix(X, self.X_train_, n_wires=self.kernel.n_wires)
        return torch.from_numpy(K_test @ self.coef_.numpy())

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "quantum_kernel_matrix",
    "FastBaseEstimator",
    "FastQuantumEstimator",
    "KernelRidge",
]
