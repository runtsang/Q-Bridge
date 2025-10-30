"""Hybrid quantum sampler and regression module.

This module combines a parameterised quantum sampler circuit (from the
SamplerQNN seed) with a quantum regression network (from the
QuantumRegression seed).  The sampler uses Qiskit primitives to produce
state‑vector samples, while the regression part is built with
torchquantum so that it can be trained end‑to‑end together with a
classical head.  The class exposes two entry points:
  * ``sample`` – returns a probability vector for the two‑qubit sampler.
  * ``forward`` – performs a regression on a batch of input states.
"""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Same data generation as the classical seed, but returning complex states."""
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
    """Dataset for quantum regression, mirroring the seed."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridQuantumSamplerRegressor(tq.QuantumModule):
    """
    Quantum sampler + regression module.

    Parameters
    ----------
    num_wires : int
        Number of qubits for the regression circuit.
    """

    def __init__(self, num_wires: int = 2):
        super().__init__()
        self.num_wires = num_wires

        # --- Sampler part (Qiskit) ------------------------------------------------
        input_params = ParameterVector("input", 2)
        weight_params = ParameterVector("weight", 4)

        qc = QuantumCircuit(2)
        qc.ry(input_params[0], 0)
        qc.ry(input_params[1], 1)
        qc.cx(0, 1)
        qc.ry(weight_params[0], 0)
        qc.ry(weight_params[1], 1)
        qc.cx(0, 1)
        qc.ry(weight_params[2], 0)
        qc.ry(weight_params[3], 1)

        self.sampler_qnn = QiskitSamplerQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params,
            sampler=StatevectorSampler(),
        )

        # --- Regression part (torchquantum) ---------------------------------------
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return probability distribution from the Qiskit sampler."""
        # Convert input tensor to numpy for Qiskit
        inputs_np = inputs.detach().cpu().numpy()
        probs = self.sampler_qnn.predict(inputs_np)
        return torch.tensor(probs, dtype=torch.float32, device=inputs.device)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Perform regression on a batch of quantum states.

        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (batch, 2**num_wires) with dtype=torch.cfloat.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)

        # Encode classical states into the quantum device
        self.encoder(qdev, state_batch)

        # Apply random layer + trainable rotations
        self.random_layer(qdev)
        for wire in range(self.num_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

        # Measurement and classical head
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["HybridQuantumSamplerRegressor", "RegressionDataset", "generate_superposition_data"]
