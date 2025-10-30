"""
Hybrid regression module (quantum).

Provides a quantum variational regression model that mirrors the classical branch.
A data generator, dataset, and fast estimator based on Qiskit StatevectorSampler are included.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import StatevectorSampler as Sampler

# ------------------------------------------------------------------
# Data generation utilities
# ------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states |ψ(θ,φ)⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩ and labels."""
    omega0 = np.zeros(2**num_wires, dtype=complex)
    omega0[0] = 1.0
    omega1 = np.zeros(2**num_wires, dtype=complex)
    omega1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1

    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset yielding complex state vectors and corresponding regression targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ------------------------------------------------------------------
# Quantum auto‑encoding layer (variational)
# ------------------------------------------------------------------
class QuantumAutoencoder(tq.QuantumModule):
    """
    Variational circuit that compresses a state into a smaller number of qubits.
    """
    def __init__(self, num_wires: int, latent_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.latent_wires = latent_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice):
        self.encoder(qdev)
        self.q_layer(qdev)
        return self.measure(qdev)

# ------------------------------------------------------------------
# Regression head
# ------------------------------------------------------------------
class RegressionHead(nn.Module):
    """
    Classical readout that maps measured qubit expectation values to a scalar.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# ------------------------------------------------------------------
# Hybrid quantum regression model
# ------------------------------------------------------------------
class HybridRegression(tq.QuantumModule):
    """
    Quantum regression model that mirrors the classical HybridRegression.
    It encodes input states, runs a variational layer, measures, and applies a
    classical readout head.
    """
    def __init__(self, num_wires: int, latent_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.latent_wires = latent_wires
        self.autoencoder = QuantumAutoencoder(num_wires, latent_wires)
        self.head = RegressionHead(num_wires)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        # Encode raw quantum data
        self.autoencoder(qdev)
        features = self.autoencoder.measure(qdev)
        return self.head(features)

# ------------------------------------------------------------------
# Fast estimator based on Qiskit
# ------------------------------------------------------------------
class FastBaseEstimator:
    """
    Evaluate a parameterised quantum circuit over many shots.
    """
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.params = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.params, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> list[list[complex]]:
        observables = list(observables)
        results: list[list[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "QuantumAutoencoder",
    "RegressionHead",
    "HybridRegression",
    "FastBaseEstimator",
]
