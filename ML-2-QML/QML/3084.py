"""FastBaseEstimator__gen007.py

Quantum version of FastBaseEstimator that supports:
- Evaluation of a Qiskit circuit via state‑vector simulation.
- Evaluation of a TorchQuantum kernel via a variational ansatz.
- Optional Gaussian shot‑noise simulation.
- A quantum RBF‑style kernel that mirrors the classical implementation.

The API mirrors the classical module so that the same experiment scripts
can be swapped between regimes.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from torchquantum.functional import func_name_dict
from typing import Iterable, List, Sequence, Callable, Optional, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class FastBaseEstimator:
    """Evaluate a quantum circuit or a TorchQuantum kernel on parameter sets."""
    def __init__(self, circuit: Union[QuantumCircuit, tq.QuantumModule]) -> None:
        self.circuit = circuit
        if isinstance(circuit, QuantumCircuit):
            self._parameters = list(circuit.parameters)
            self._is_circuit = True
        else:
            self._is_circuit = False

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a bound copy of the Qiskit circuit for the given parameters."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Return a list of observable expectation values for each parameter set.

        When ``shots`` is supplied, the result is perturbed with Gaussian noise
        mimicking measurement statistics.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if self._is_circuit:
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
        else:  # TorchQuantum kernel
            for values in parameter_sets:
                val = self.circuit.forward(
                    torch.tensor([values], dtype=torch.float32),
                    torch.tensor([values], dtype=torch.float32)
                ).item()
                results.append([complex(val)])

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    # The evaluate method is inherited and already adds noise when ``shots`` is not ``None``.
    pass

# ----------------------------------------------------------------------
# Quantum kernel utilities (TorchQuantum)
# ----------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data into a variational circuit."""
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
    """Quantum RBF‑style kernel evaluated via a fixed TorchQuantum ansatz."""
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
    """Compute the Gram matrix between ``a`` and ``b`` using the quantum kernel."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["FastBaseEstimator", "FastEstimator", "KernalAnsatz", "Kernel", "kernel_matrix"]
