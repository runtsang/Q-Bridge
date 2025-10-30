from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN as QuantumSamplerQNN

class QuantumEncoder(nn.Module):
    """A lightweight quantum encoder using Qiskit's RealAmplitudes ansatz."""
    def __init__(self, input_dim: int, latent_dim: int, reps: int = 3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reps = reps
        self.n_qubits = max(input_dim, int(np.ceil(np.log2(latent_dim))))
        self.ansatz = RealAmplitudes(self.n_qubits, reps=reps)
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.append(self.ansatz, range(self.n_qubits))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        padded = torch.zeros(batch_size, self.n_qubits, device=inputs.device)
        padded[:, :self.input_dim] = inputs
        results = []
        for i in range(batch_size):
            vec = padded[i].cpu().numpy()
            circ = self.circuit.copy()
            circ.set_parameters(vec, range(self.n_qubits))
            state = Statevector(circ).data
            amplitudes = np.abs(state[:self.latent_dim])
            results.append(amplitudes)
        return torch.tensor(results, dtype=torch.float32, device=inputs.device)

class SamplerModule(nn.Module):
    """Classical sampler network mirroring a quantum SamplerQNN."""
    def __init__(self, input_dim: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder combining classical encoder and optional quantum encoder."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: tuple[int,...] = (128, 64),
        dropout: float = 0.1,
        use_quantum_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.use_quantum_encoder = use_quantum_encoder
        if use_quantum_encoder:
            self.encoder = QuantumEncoder(input_dim, latent_dim)
        else:
            layers = []
            in_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                in_dim = h
            layers.append(nn.Linear(in_dim, latent_dim))
            self.encoder = nn.Sequential(*layers)

        layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for the hybrid autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = ["HybridAutoencoder", "train_hybrid_autoencoder"]
