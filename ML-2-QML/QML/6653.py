"""Quantum regression model built with Qiskit that shares the same API as the classical MLP.

Key components:
- Data generation creates complex superposition states.
- A regression dataset that yields state vectors and targets.
- EstimatorQNN__gen501: a Qiskit EstimatorQNN wrapping a parameterised circuit that
  encodes inputs, applies a trainable random SU(2) layer, and measures Pauli‑Z.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import EfficientSU2

__all__ = ["EstimatorQNN__gen501", "RegressionDataset", "generate_superposition_data"]


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the circuit.
    samples : int
        Number of examples to generate.

    Returns
    -------
    states : np.ndarray, shape (samples, 2**num_wires)
        Complex amplitudes of each state.
    labels : np.ndarray, shape (samples,)
        Regression targets derived from the angles.
    """
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
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that yields complex state vectors and real targets."""

    def __init__(self, samples: int, num_wires: int = 2):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class EstimatorQNN__gen501:
    """
    Qiskit EstimatorQNN wrapper that mirrors the classical MLP interface.

    The circuit consists of:
        1. Ry rotation encoding of each input feature.
        2. A trainable random SU(2) layer (EfficientSU2).
        3. Measurement of Pauli‑Z on all qubits.
    """

    def __init__(self, num_wires: int = 2, weight_init: float = 0.0) -> None:
        # Define trainable parameters
        self.input_params = [Parameter(f"inp_{i}") for i in range(num_wires)]
        self.weight_params = [Parameter(f"w_{i}") for i in range(num_wires)]

        # Build circuit
        self.circuit = QuantumCircuit(num_wires)
        for i, p in enumerate(self.input_params):
            self.circuit.ry(p, i)
        for i, p in enumerate(self.weight_params):
            self.circuit.rx(p, i)

        # Add a random efficient SU(2) layer for expressivity
        self.circuit.append(EfficientSU2(num_wires, reps=1), range(num_wires))

        # Observable: Pauli‑Z on all qubits (sum)
        self.observable = SparsePauliOp.from_list([("Z" * num_wires, 1.0)])

        # Estimator primitive
        self.estimator = StatevectorEstimator()

        # Wrap into EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Execute the quantum circuit for a batch of input states.

        Parameters
        ----------
        state_batch : torch.Tensor, shape (batch, num_wires)
            Real input features that are encoded via Ry gates.

        Returns
        -------
        torch.Tensor, shape (batch,)
            Expected value of the observable for each input.
        """
        # Convert states to numpy for Qiskit
        inp_np = state_batch.cpu().numpy()
        # Qiskit expects a list of ParameterBindings
        param_dicts = [{p: v for p, v in zip(self.input_params, inp_np[i])} for i in range(inp_np.shape[0])]
        # Run estimator
        expectation = self.estimator.run(
            circuits=[self.circuit],  # single circuit reused
            parameter_binds=param_dicts,
            observables=[self.observable],
        ).result().values.squeeze()

        return torch.tensor(expectation, dtype=torch.float32, device=state_batch.device)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that forwards without gradient."""
        with torch.no_grad():
            return self.forward(x)
