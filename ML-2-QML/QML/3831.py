"""Hybrid estimator that unifies quantum circuit evaluation with a classical head.

The class accepts a parametrized QuantumCircuit, an optional linear head,
and evaluates expectation values of Pauli operators.  It can also
apply a classical regression head to the measured expectation values,
mimicking the quantum regression demo.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Callable[[complex], complex]  # For quantum expectation values


# --------------------------------------------------------------------------- #
# Dataset and model utilities copied from QuantumRegression.py
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a superposition of |0…0⟩ and |1…1⟩ with random phases."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):
        return {"states": torch.tensor(self.states[idx], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}


class QModel(nn.Module):
    """Linear head applied to expectation values of all Pauli‑Z operators."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.head = nn.Linear(num_wires, 1)

    def forward(self, expectations: torch.Tensor) -> torch.Tensor:
        return self.head(expectations).squeeze(-1)


# --------------------------------------------------------------------------- #
# Hybrid quantum estimator implementation
# --------------------------------------------------------------------------- #
class FastHybridEstimator:
    """Hybrid estimator that evaluates a parametrized circuit and optionally a classical head.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrized circuit whose parameters are bound per evaluation call.
    head : nn.Module | None
        Optional PyTorch linear head that transforms the vector of expectation
        values into a scalar output.  When ``None`` the raw expectation values
        are returned.
    """

    def __init__(self, circuit: QuantumCircuit, head: nn.Module | None = None) -> None:
        self.circuit = circuit
        self.head = head
        self.parameters = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of BaseOperator objects whose expectation values are to be measured.
        parameter_sets
            Sequence of parameter lists to evaluate.
        shots
            If provided, the expectation values are sampled using a shot‑based
            simulator with the specified number of shots.  Otherwise a state‑vector
            simulator is used.
        seed
            Seed for the shot simulator.
        """
        results: List[List[float]] = []

        for params in parameter_sets:
            circuit = self._bind(params)
            if shots is None:
                sv = Statevector.from_instruction(circuit)
                row = [float(sv.expectation_value(obs)) for obs in observables]
            else:
                # Simple shot simulator via Statevector sampling
                sv = Statevector.from_instruction(circuit)
                row = []
                rng = np.random.default_rng(seed)
                for obs in observables:
                    exp_val = float(sv.expectation_value(obs))
                    probs = np.array([(1 - exp_val) / 2, (1 + exp_val) / 2])
                    outcomes = rng.choice([1, -1], size=shots, p=probs)
                    row.append(outcomes.mean())
            results.append(row)

        if self.head is None:
            return results

        # Convert to tensor for the head
        tensor = torch.tensor(results, dtype=torch.float32)
        out = self.head(tensor).squeeze(-1).cpu().tolist()
        # Wrap each scalar in a list to preserve 2‑D shape
        return [[v] for v in out]


__all__ = ["FastHybridEstimator", "RegressionDataset", "generate_superposition_data", "QModel"]
