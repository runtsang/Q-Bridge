"""Quantum regression model with a hybrid estimator and fully‑connected layer."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torchquantum as tq
from typing import Iterable, Sequence, List, Callable
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit import QuantumCircuit

# Data generation
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states |ψ(θ,φ)⟩ = cosθ|0⟩+e^{iφ}sinθ|1⟩ and labels
    y = sin(2θ)cosφ.  The function is vectorised for speed.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    states = np.cos(thetas)[:, None] * omega_0 + np.exp(1j * phis)[:, None] * np.sin(thetas)[:, None] * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)

# Dataset
class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns a dictionary with ``states`` and ``target``."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# Hybrid layer inspired by FCL
class FullyConnectedLayer(tq.QuantumModule):
    """A parameterised quantum circuit that emulates a fully‑connected layer
    via a sequence of RX/RY gates on each qubit followed by a measurement.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
        return self.measure(qdev)

# Quantum model
class QModel(tq.QuantumModule):
    """A hybrid quantum‑classical regression model that combines a
    data‑encoding layer, a random feature layer, a fully‑connected quantum
    layer, and a classical read‑out head.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder: simple Ry rotations for each qubit
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        # Random layer to generate a rich feature map
        self.random_layer = tq.RandomLayer(
            n_ops=30, wires=list(range(num_wires)), has_params=False
        )
        # Fully connected quantum layer
        self.fcl = FullyConnectedLayer(num_wires)
        # Classical read‑out
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.random_layer(qdev)
        features = self.fcl(qdev)
        return self.head(features).squeeze(-1)

# Estimator utilities
class FastBaseEstimator:
    """Fast deterministic estimator for a quantum circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self._params, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            results.append([state.expectation_value(obs) for obs in observables])
        return results

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
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
            noisy.append([rng.normal(c.real, max(1e-6, 1 / shots)) + 1j * rng.normal(c.imag, max(1e-6, 1 / shots)) for c in row])
        return noisy

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data",
           "FullyConnectedLayer", "FastBaseEstimator", "FastEstimator"]
