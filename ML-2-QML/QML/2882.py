"""Hybrid quantum regression module with quantum‑classical interface.

The module contains:
* A dataset that produces superposition states and the associated target.
* A variational circuit built with torchquantum that encodes input angles,
  applies a random layer and trainable RX/RY rotations, measures Pauli‑Z,
  and maps the expectation values to a scalar output.
* A FastEstimator that evaluates a qiskit circuit for a set of parameters
  and adds Gaussian shot noise to emulate finite‑shot measurements.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, Sequence, List, Tuple

# --------------------------------------------------------------------------- #
# Data generation utilities
# --------------------------------------------------------------------------- #

def _generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    The angles theta and phi are used as classical features for the dataset.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    states = np.empty((samples, 2 ** num_wires), dtype=complex)
    for i, (theta, phi) in enumerate(zip(thetas, phis)):
        states[i] = np.cos(theta) * omega_0 + np.exp(1j * phi) * np.sin(theta) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class HybridRegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns a quantum state vector, the target, and
    the underlying classical angles for debugging or hybrid training.
    """

    def __init__(self, samples: int, num_wires: int = 3):
        self.states, self.labels = _generate_superposition_data(num_wires, samples)
        self.num_wires = num_wires

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Quantum model
# --------------------------------------------------------------------------- #

class HybridRegression(tq.QuantumModule):
    """
    Variational circuit for regression.

    Architecture:
        * General encoder that maps the input state to the device.
        * RandomLayer (30 ops) + trainable RX/RY per wire.
        * Measure all wires in Pauli‑Z basis.
        * Linear head mapping the expectation values to a scalar.
    """

    class _QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int = 3):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = self._QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
# Estimator utilities
# --------------------------------------------------------------------------- #

class FastEstimator:
    """
    Evaluate a parameterised qiskit circuit for a set of parameters and
    observables, optionally adding Gaussian shot noise.
    """

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.
        If *shots* is provided, Gaussian noise with variance 1/shots is added
        to each expectation value to emulate finite‑shot sampling.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            state = Statevector.from_instruction(self._bind(params))
            row = [state.expectation_value(obs) for obs in observables]
            if shots is not None:
                rng = np.random.default_rng(seed)
                noisy_row = [
                    rng.normal(float(val.real), max(1e-6, 1 / shots)) + 1j * rng.normal(float(val.imag), max(1e-6, 1 / shots))
                    for val in row
                ]
                row = noisy_row
            results.append(row)
        return results

__all__ = ["HybridRegression", "HybridRegressionDataset", "FastEstimator"]
