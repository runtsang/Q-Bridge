"""
HybridAutoEncoder: classical encoder/decoder + fixed quantum latent layer.

The module is a drop‑in replacement for the original Autoencoder factory.
It keeps the same public API but internally stitches together:
* a PyTorch MLP encoder / decoder (identical to the seed's AutoencoderNet)
* a deterministic quantum circuit that maps the classical latent vector to a
  statevector and returns the expectation values of Pauli‑Z on each qubit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable, Tuple, List

# Quantum simulator – we keep the quantum part entirely deterministic
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector

# --------------------------------------------------------------------------- #
#   Classical autoencoder backbone
# --------------------------------------------------------------------------- #

@dataclass
class HybridAutoencoderConfig:
    """Configuration for the hybrid network."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class ClassicalAutoencoderNet(nn.Module):
    """Same as the seed's AutoencoderNet but renamed for clarity."""
    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

# --------------------------------------------------------------------------- #
#   Fixed quantum latent layer
# --------------------------------------------------------------------------- #

class FixedQuantumLatent(nn.Module):
    """Deterministic quantum circuit that turns a latent vector into a statevector
    and returns the expectation values of Pauli‑Z on each qubit.
    The circuit contains no trainable parameters.
    """
    def __init__(self, latent_dim: int, num_trash: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.sim = AerSimulator(method="statevector")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return a classical vector derived from the quantum state."""
        batch_size = z.shape[0]
        outputs: List[torch.Tensor] = []
        for i in range(batch_size):
            qc = QuantumCircuit(self.latent_dim + 2 * self.num_trash + 1)
            # Encode the latent vector into RX rotations
            for q in range(self.latent_dim):
                angle = float(z[i, q].item() * 2 * torch.pi)
                qc.rx(angle, q)
            # Simple entanglement with trash qubits
            for q in range(self.latent_dim + self.num_trash):
                qc.cx(q, q + self.num_trash)
            # Auxiliary Hadamard for interference
            qc.h(self.latent_dim + 2 * self.num_trash)
            # Simulate
            job = execute(qc, self.sim)
            state = Statevector(job.result().get_statevector(qc))
            exp_z = torch.tensor(
                [state.expectation_value('Z' * (q + 1)).real for q in range(self.latent_dim)],
                dtype=torch.float32,
            )
            outputs.append(exp_z)
        return torch.stack(outputs, dim=0)

# --------------------------------------------------------------------------- #
#   Hybrid autoencoder
# --------------------------------------------------------------------------- #

class HybridAutoEncoder(nn.Module):
    """Hybrid autoencoder that uses a classical encoder/decoder and a fixed
    quantum latent layer.
    """
    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.classical = ClassicalAutoencoderNet(cfg)
        self.quantum = FixedQuantumLatent(cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.classical.encode(x)
        qz = self.quantum(z)
        recon = self.classical.decode(qz)
        return recon

# --------------------------------------------------------------------------- #
#   Factory & training utilities
# --------------------------------------------------------------------------- #

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoEncoder:
    """Factory that mirrors the quantum helper returning a configured network."""
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoEncoder(cfg)

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

def train_hybrid_autoencoder(
    model: HybridAutoEncoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "Autoencoder",
    "HybridAutoencoderConfig",
    "HybridAutoEncoder",
    "train_hybrid_autoencoder",
]
