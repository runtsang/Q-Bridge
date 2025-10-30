from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.quantum_info import Statevector, Pauli
from qiskit.primitives import Sampler as QiskitSampler

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 8
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    gamma: float = 1.0
    kernel_weight: float = 0.01

class QuantumEncoder(nn.Module):
    def __init__(self, num_qubits: int, latent_dim: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.feature_circuit = RawFeatureVector(num_qubits)
        self.ansatz = RealAmplitudes(num_qubits, reps=3)
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit.append(self.feature_circuit, range(num_qubits))
        self.circuit.append(self.ansatz, range(num_qubits))
        self.sampler = QiskitSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.feature_circuit.parameters,
            weight_params=self.ansatz.parameters,
            interpret=self._interpret,
            output_shape=2,
            sampler=self.sampler,
        )

    def _interpret(self, x: torch.Tensor) -> torch.Tensor:
        sv = Statevector(x, dims=(2,) * self.num_qubits)
        exps = []
        for i in range(self.num_qubits):
            exps.append(sv.expectation_value(Pauli('Z'), qubits=[i]).real)
        return torch.tensor(exps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.qnn(x)
        if latent.shape[-1] > self.latent_dim:
            latent = latent[:, :self.latent_dim]
        return latent

class HybridAutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = QuantumEncoder(config.latent_dim, config.latent_dim)
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.kernel = Kernel(config.gamma)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 8,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    gamma: float = 1.0,
    kernel_weight: float = 0.01,
) -> HybridAutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        gamma=gamma,
        kernel_weight=kernel_weight,
    )
    return HybridAutoencoderNet(config)

def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            latent = model.encode(batch)
            reconstruction = model.decode(latent)
            recon_loss = loss_fn(reconstruction, batch)
            kernel_loss = -torch.mean(model.kernel(latent, batch))
            loss = recon_loss + model.config.kernel_weight * kernel_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridAutoencoder", "HybridAutoencoderNet", "AutoencoderConfig", "train_hybrid_autoencoder"]
