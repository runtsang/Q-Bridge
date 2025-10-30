"""FastBaseEstimator: a quantum estimator for parameterised circuits.

Extends the original Qiskit implementation by:
- Supporting optional shot noise using a simple Gaussian model.
- Providing a synthetic dataset for quantum regression.
- Including a small variational circuit that can be trained for regression.

Features
--------
* Deterministic expectation value calculation via Statevector.
* Optional shot‑noise simulation.
* Dataset and model for regression on superposition‑like states.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional, Sequence, List, Union

import numpy as np
import torch
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]  # alias for consistency


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Same as in the quantum reference."""
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
    """Dataset for quantum regression."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel:
    """Variational circuit that can be used for regression."""

    def __init__(self, num_wires: int):
        self.num_wires = num_wires
        self.circuit = QuantumCircuit(num_wires)
        # Simple encoding: use RX with parameter to encode state
        for i in range(num_wires):
            self.circuit.rx(0.0, i)
        # Random layer
        for _ in range(3):
            for i in range(num_wires):
                self.circuit.rz(np.pi / 4, i)
                self.circuit.rx(np.pi / 2, i)
        # Trainable RX/RY layer
        for i in range(num_wires):
            self.circuit.rx(0.0, i)
            self.circuit.ry(0.0, i)

    def bind_parameters(self, params: Sequence[float]) -> QuantumCircuit:
        """Return a copy of the circuit with parameters bound."""
        if len(params)!= self.num_wires * 2:
            raise ValueError("Parameter list length must equal twice the number of wires.")
        bound = self.circuit.copy()
        for i, param in enumerate(params[:self.num_wires]):
            bound.rx(param, i)
        for i, param in enumerate(params[self.num_wires:]):
            bound.ry(param, i)
        return bound


class FastBaseEstimator:
    """Quantum estimator with optional shot noise."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        self.parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def _expectation(self, circuit: QuantumCircuit, observable: BaseOperator) -> complex:
        state = Statevector.from_instruction(circuit)
        return state.expectation_value(observable)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        rng = np.random.default_rng(self.seed)

        for params in parameter_sets:
            bound = self._bind(params)
            row: List[complex] = []
            for obs in observables:
                exp_val = self._expectation(bound, obs)
                if self.shots is not None:
                    # Add Gaussian shot noise with std 1/sqrt(shots)
                    noise = rng.normal(0, 1 / np.sqrt(self.shots))
                    exp_val = exp_val + noise
                row.append(exp_val)
            results.append(row)
        return results


__all__ = [
    "FastBaseEstimator",
    "RegressionDataset",
    "QModel",
    "generate_superposition_data",
]
