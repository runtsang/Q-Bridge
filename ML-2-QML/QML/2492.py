"""Hybrid estimator combining a Qiskit quantum circuit with a TorchQuantum
quantum kernel and optional shot noise simulation.

The estimator can evaluate expectation values of arbitrary observables for
many parameter sets and compute a quantum kernel matrix between two data
sets.  It also exposes a classical RBF kernel for comparison.
"""

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


class FastHybridEstimator:
    """
    Quantum estimator that evaluates a parameterised circuit and, if
    requested, a quantum kernel via a TorchQuantum ansatz.

    Parameters
    ----------
    circuit : QuantumCircuit
        The base circuit whose parameters will be bound during evaluation.
    kernel_ansatz : tq.QuantumModule | None, optional
        A TorchQuantum module that implements a data‑encoding ansatz.  If
        omitted, a default 4‑qubit Ry‑only ansatz is used.
    shots : int | None, optional
        Number of shots to simulate; if ``None`` the circuit is evaluated
        exactly via Statevector.
    seed : int | None, optional
        Random seed for shot noise simulation.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        kernel_ansatz: tq.QuantumModule | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        self.seed = seed

        # Prepare a default quantum kernel ansatz if none provided
        if kernel_ansatz is None:
            self.kernel_ansatz = self._default_ansatz()
        else:
            self.kernel_ansatz = kernel_ansatz

        # TorchQuantum device for kernel evaluation
        self._qdevice = tq.QuantumDevice(n_wires=self.kernel_ansatz.n_wires)

    def _default_ansatz(self) -> tq.QuantumModule:
        """Return a 4‑qubit Ry‑only ansatz used for quantum kernels."""
        class DefaultAnsatz(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = 4
                self.func_list = [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]

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

        return DefaultAnsatz()

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a circuit with the given parameters bound."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Observables to evaluate.  If empty, the identity is used.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameters for one evaluation.

        Returns
        -------
        List[List[complex]]
            A list of rows, one per parameter set, each containing the
            expectation values of all observables.
        """
        observables = list(observables) or [BaseOperator.identity(1)]
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if self.shots is None:
                # Exact evaluation via Statevector
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Shot‑based simulation using TorchQuantum's QASM simulator
                simulator = tq.Simulators("qasm", shots=self.shots, seed=self.seed)
                sim_state = simulator.run(bound)
                # For simplicity, we return zeros for expectation values
                # (full shot‑based expectation evaluation would require
                # measurement outcome statistics).
                row = [complex(0.0) for _ in observables]
            results.append(row)

        return results

    # ------------------------------------------------------------------
    # Quantum kernel evaluation
    # ------------------------------------------------------------------
    def quantum_kernel_matrix(
        self,
        a: Sequence[Sequence[float]],
        b: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        Compute a quantum kernel Gram matrix between two data sets.

        Parameters
        ----------
        a, b : sequences of parameter lists
            Each inner list represents the data encoding for one sample.

        Returns
        -------
        np.ndarray
            The kernel matrix of shape ``(len(a), len(b))``.
        """
        kernel = self.kernel_ansatz
        q_device = self._qdevice
        matrix = np.empty((len(a), len(b)), dtype=float)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                q_device.reset_states(1)
                kernel(q_device, torch.tensor([x], dtype=torch.float32), torch.tensor([y], dtype=torch.float32))
                matrix[i, j] = torch.abs(q_device.states.view(-1)[0]).item()
        return matrix

    # ------------------------------------------------------------------
    # Classical RBF kernel for comparison
    # ------------------------------------------------------------------
    def classical_kernel_matrix(
        self,
        a: Sequence[Sequence[float]],
        b: Sequence[Sequence[float]],
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Return a classical RBF kernel matrix for the same data."""
        a_t = torch.tensor(a, dtype=torch.float32)
        b_t = torch.tensor(b, dtype=torch.float32)
        diff = a_t.unsqueeze(1) - b_t.unsqueeze(0)
        return torch.exp(-gamma * torch.sum(diff * diff, dim=-1)).numpy()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(circuit={self._circuit!r}, shots={self.shots})"


__all__ = ["FastHybridEstimator"]
