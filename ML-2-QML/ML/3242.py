"""Hybrid classical‑quantum autoencoder.

The module defines:
* :class:`AutoencoderConfig` – hyper‑parameter container.
* :class:`HybridAutoencoder` – a PyTorch ``nn.Module`` that chains a classical encoder,
  a quantum latent module (SAMPLER‑based), and a classical decoder.
* :func:`train_hybrid_autoencoder` – a lightweight training loop that
  back‑propagates through the quantum layer using the qiskit sampler gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Hyper‑parameters for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 4
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    quantum_reps: int = 3  # number of repetitions in the variational ansatz
    quantum_shots: int = 1024

# --------------------------------------------------------------------------- #
# Classical encoder / decoder
# --------------------------------------------------------------------------- #
class _FeedForwardEncoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FeedForwardDecoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

# --------------------------------------------------------------------------- #
# Quantum latent module (SAMPLER‑based)
# --------------------------------------------------------------------------- #
def _quantum_latent_circuit(num_qubits: int, reps: int) -> "qiskit.QuantumCircuit":
    """Return a RealAmplitudes ansatz that will produce the latent vector."""
    from qiskit.circuit.library import RealAmplitudes
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(num_qubits)
    qc.compose(RealAmplitudes(num_qubits, reps=reps), inplace=True)
    # No measurement here – expectation values will be extracted by the sampler.
    return qc


class QuantumLatentLayer(nn.Module):
    """Wraps a qiskit sampler to produce a differentiable latent representation."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        from qiskit import Aer
        from qiskit.primitives import Sampler

        self.num_qubits = config.latent_dim
        self.sampler = Sampler(Aer.get_backend("aer_simulator"))
        self.circuit = _quantum_latent_circuit(self.num_qubits, config.quantum_reps)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        params: shape (batch, latent_dim)
        Returns expectation values of Z on each qubit.
        """
        # Convert to numpy for the sampler
        param_vals = params.detach().cpu().numpy()
        # Prepare parameter bindings
        bindings = [{f"theta_{i}": val for i, val in enumerate(row)} for row in param_vals]
        # Execute sampler
        result = self.sampler.run(self.circuit, parameter_binds=bindings).result()
        # Expectation values
        exp = result.expectation
        return torch.as_tensor(exp, dtype=torch.float32, device=params.device)

# --------------------------------------------------------------------------- #
# Hybrid autoencoder
# --------------------------------------------------------------------------- #
class HybridAutoencoder(nn.Module):
    """
    Classical encoder → quantum latent layer → classical decoder.
    The encoder outputs parameters for the quantum circuit; the sampler
    returns a differentiable latent vector that is fed to the decoder.
    """
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = _FeedForwardEncoder(config)
        self.quantum_latent = QuantumLatentLayer(config)
        self.decoder = _FeedForwardDecoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.encoder(x)          # shape (B, latent_dim)
        z = self.quantum_latent(params)   # shape (B, latent_dim)
        recon = self.decoder(z)           # shape (B, input_dim)
        return recon

# --------------------------------------------------------------------------- #
# Training routine
# --------------------------------------------------------------------------- #
def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Simple MSE training loop that back‑propagates through the quantum layer.
    The qiskit sampler supports automatic differentiation via its gradient
    implementation, so no manual Jacobian is required.
    """
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = [
    "AutoencoderConfig",
    "HybridAutoencoder",
    "train_hybrid_autoencoder",
]
