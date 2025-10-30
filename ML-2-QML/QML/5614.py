"""Quantum regression model that mirrors the classical hybrid implementation.

The quantum implementation uses torchquantum for efficient simulation and
qiskit for a sampler circuit.  It follows the same public API as the
classical `HybridRegressor` while replacing each stand‑in layer with a
quantum equivalent.  The design preserves the API so the two modules
can be swapped without changing downstream code.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import qiskit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler as Sampler

# ----------------------------------------------------------------------
# Data generation – identical to the classical seed
# ----------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
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

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ----------------------------------------------------------------------
# Quantum stand‑ins for classical layers
# ----------------------------------------------------------------------
class QuantumConvLayer(tq.QuantumModule):
    """A simple quantum convolution style layer using a random circuit."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.random = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice):
        self.encoder(qdev)
        self.random(qdev)
        return self.measure(qdev)

class QuantumFCLayer(tq.QuantumModule):
    """A quantum fully‑connected layer implemented with a RandomLayer."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice):
        self.random(qdev)
        return self.measure(qdev)

class QuantumSampler(tq.QuantumModule):
    """A quantum sampler that uses qiskit’s StatevectorSampler."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # build a simple parameterised circuit
        self.circuit = qiskit.QuantumCircuit(num_wires)
        self.params = ParameterVector("theta", num_wires)
        for i in range(num_wires):
            self.circuit.ry(self.params[i], i)
        self.circuit.measure_all()
        self.sampler = Sampler()
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def forward(self, qdev: tq.QuantumDevice):
        # sample from the circuit using the state of the device
        # (for simplicity we ignore the device state and sample fresh)
        job = qiskit.execute(self.circuit, self.backend, shots=100)
        result = job.result().get_counts(self.circuit)
        probs = np.array([v / 100 for v in result.values()])
        return probs

# ----------------------------------------------------------------------
# Quantum regression model
# ----------------------------------------------------------------------
class HybridRegressor(tq.QuantumModule):
    """
    Quantum implementation of the hybrid regression model.  It mirrors the
    classical `HybridRegressor` but replaces each stand‑in layer with a
    quantum equivalent.  The public API remains identical so the two
    modules can be swapped without changing downstream code.
    """

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.conv = QuantumConvLayer(num_wires)
        self.fcl = QuantumFCLayer(num_wires)
        self.sampler = QuantumSampler(num_wires)
        # total features: conv + fcl + sampler mean
        self.head = nn.Linear(2 * num_wires + 1, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of quantum states with shape (batch, 2**num_wires).

        Returns
        -------
        torch.Tensor
            Predicted scalar for each sample in the batch.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the classical data into the quantum device
        conv_features = self.conv(qdev)
        fcl_features = self.fcl(qdev)
        sampler_out = self.sampler(qdev)
        # Convert sampler output to a tensor and compute mean
        sampler_mean = torch.tensor(sampler_out).float().mean().unsqueeze(0)
        # Expand to batch dimension
        sampler_mean = sampler_mean.expand(bsz, -1)
        # Concatenate all features
        features = torch.cat([conv_features, fcl_features, sampler_mean], dim=-1)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressor", "RegressionDataset", "generate_superposition_data"]
