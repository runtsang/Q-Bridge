"""Hybrid autoencoder combining classical MLP with a variational quantum latent layer.

The model encodes the input, transforms the latent representation with a
parameterized quantum circuit, and decodes back to the original space.
Training alternates classical gradient descent on the MLP and a COBYLA
gradient‑free optimizer on the quantum parameters.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Iterable, Tuple, List

# Utility to convert input to tensor
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

# Import the quantum latent layer
from.quantum_autoencoder import QuantumLatentLayer

# ----------------------------------------------------------------------
# Classical utilities
# ----------------------------------------------------------------------
class FCL(nn.Module):
    """A lightweight fully‑connected layer that mimics a quantum FCL."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values)).mean(dim=0).detach().numpy()

# ----------------------------------------------------------------------
# Estimator for quantum circuits
# ----------------------------------------------------------------------
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit import QuantumCircuit

class FastBaseEstimator:
    """Evaluate expectation values of a parametrized quantum circuit."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Iterable[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            results.append([state.expectation_value(obs) for obs in observables])
        return results

# ----------------------------------------------------------------------
# Configuration and core network
# ----------------------------------------------------------------------
class AutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int
    hidden_dims: Tuple[int,...]
    dropout: float
    quantum_reps: int
    quantum_shots: int

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int,...] = (128, 64),
        dropout: float = 0.1,
        quantum_reps: int = 2,
        quantum_shots: int = 1024,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.quantum_reps = quantum_reps
        self.quantum_shots = quantum_shots

class AutoencoderNet(nn.Module):
    """Purely classical autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder.append(nn.Linear(in_dim, h))
            encoder.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder.append(nn.Dropout(config.dropout))
            in_dim = h
        encoder.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder.append(nn.Linear(in_dim, h))
            decoder.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder.append(nn.Dropout(config.dropout))
            in_dim = h
        decoder.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class HybridAutoencoder(nn.Module):
    """Autoencoder with a quantum latent transformation."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.classical = AutoencoderNet(config)
        self.quantum = QuantumLatentLayer(
            latent_dim=config.latent_dim,
            reps=config.quantum_reps,
            shots=config.quantum_shots,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.classical.encode(x)
        z_q = self.quantum(z.cpu().numpy())
        return self.classical.decode(torch.as_tensor(z_q, dtype=torch.float32, device=x.device))

# ----------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------
def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Train the hybrid autoencoder.

    Classical parameters are updated with Adam.
    Quantum parameters are optimized with COBYLA after each epoch.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Classical optimizer
    classical_params = [
        p for name, p in model.named_parameters() if "quantum" not in name
    ]
    opt = torch.optim.Adam(classical_params, lr=lr, weight_decay=weight_decay)

    # Quantum optimizer
    from qiskit_machine_learning.optimizers import COBYLA

    def quantum_loss(params: List[float]) -> float:
        model.quantum.set_weight_params(params)
        total = 0.0
        for batch, in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = nn.functional.mse_loss(recon, batch, reduction="sum")
            total += loss.item()
        return total / len(dataset)

    history: List[float] = []

    for epoch in range(epochs):
        # Classical training
        model.train()
        for batch, in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = nn.functional.mse_loss(recon, batch)
            loss.backward()
            opt.step()

        # Quantum optimisation
        init = list(model.quantum.weight_params)
        res = COBYLA(quantum_loss, init, tol=1e-3).optimize()
        model.quantum.set_weight_params(res.x)

        # Record epoch loss
        model.eval()
        with torch.no_grad():
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(device)
                recon = model(batch)
                epoch_loss += nn.functional.mse_loss(recon, batch, reduction="sum").item()
            epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history

__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "HybridAutoencoder",
    "train_hybrid_autoencoder",
    "FCL",
    "FastBaseEstimator",
]
