"""Hybrid quantum kernel module with a TorchQuantum ansatz.

The :class:`Kernel` class is a ``torchquantum.QuantumModule`` that
encodes two classical feature vectors into a quantum state and
returns the overlap of the resulting states.  The implementation
mirrors the original seed but adds a few conveniences:

* ``n_wires`` and ``func_list`` are now optional keyword arguments,
  enabling the kernel to be configured at construction time.
* ``kernel_matrix`` accepts either ``torch.Tensor`` or
  ``Sequence[Sequence[float]]``.
* ``FastBaseEstimator`` is extended to support shot noise via the
  ``shots`` keyword, matching the classical counterpart.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum import Statevector


class Kernel(tq.QuantumModule):
    """Quantum kernel based on a parameter‑shaped ansatz.

    Parameters
    ----------
    n_wires:
        Number of qubits used by the device.
    func_list:
        Optional list of gate specifications.  If omitted the
        default is a single‑qubit ``ry`` rotation on each wire.
    """

    def __init__(
        self,
        n_wires: int = 4,
        *,
        func_list: Optional[List[dict]] = None,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        if func_list is None:
            # Default to a simple RY rotation per qubit
            func_list = [
                {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)
            ]
        self.ansatz = KernalAnsatz(func_list)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for two feature vectors."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Compute the Gram matrix for two collections of vectors."""
        return np.array([[self(x, y).item() for y in b] for x in a])


class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""

    def __init__(self, func_list: List[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""

    def __init__(self, circuit: tq.QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> tq.QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[tq.QuantumOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables:
            Quantum operators whose expectation values are to be computed.
        parameter_sets:
            A sequence of parameter vectors to evaluate.
        shots:
            If provided, the expectation values are sampled with Gaussian
            noise whose standard deviation is ``1 / sqrt(shots)``.
        seed:
            Random seed for reproducibility.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circuit = self._bind(values)
            state = Statevector.from_instruction(circuit)
            row = [state.expectation_value(obs) for obs in observables]
            if shots is not None:
                rng = np.random.default_rng(seed)
                noise = rng.normal(0, 1 / np.sqrt(shots), size=len(row))
                row = [val + noise[i] for i, val in enumerate(row)]
            results.append(row)
        return results


__all__ = [
    "Kernel",
    "KernalAnsatz",
    "FastBaseEstimator",
]
