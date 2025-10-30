"""Hybrid quantum autoencoder using Qiskit and TorchQuantum for regression."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

def _domain_wall(circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
    """Apply a domain wall (X gates) on qubits a … b‑1."""
    for i in range(a, b):
        circuit.x(i)
    return circuit

def _auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Build a variational auto‑encoder circuit with swap‑test reconstruction."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encoder ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Swap‑test for reconstruction
    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

def QuantumAutoencoder(
    latent_dim: int,
    num_trash: int = 2,
    *,
    device: str | None = None,
) -> SamplerQNN:
    """Factory that returns a quantum auto‑encoder implemented as a SamplerQNN."""
    algorithm_globals.random_seed = 42
    sampler = StatevectorSampler(device=device)

    circuit = _auto_encoder_circuit(latent_dim, num_trash)
    # Optionally add a domain wall before the auto‑encoder
    dw_circuit = _domain_wall(QuantumCircuit(latent_dim + 2 * num_trash), 0, latent_dim + 2 * num_trash)
    circuit.compose(dw_circuit, range(0, latent_dim + 2 * num_trash), inplace=True)

    def identity_interpret(x):
        return x

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

def train_quantum_autoencoder(
    model: SamplerQNN,
    dataloader: torch.utils.data.DataLoader,
    *,
    epochs: int = 50,
    lr: float = 1e-2,
    device: str | None = None,
) -> list[float]:
    """Train the quantum auto‑encoder using a simple MSE loss on the measurement outcome."""
    device = device or "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            inputs = batch["states"].to(device)
            targets = batch["target"].unsqueeze(-1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.shape[0]
        epoch_loss /= len(dataloader.dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "QuantumAutoencoder",
    "RegressionDataset",
    "generate_superposition_data",
    "train_quantum_autoencoder",
]
