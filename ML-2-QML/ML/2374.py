# Autoencoder__gen178.py - Classical part
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

from dataclasses import dataclass
from typing import Tuple, Iterable, List, Optional

# ----------------------------------------------------------------------
# Data utilities – mix of classical and quantum data generation
# ----------------------------------------------------------------------
def generate_classical_data(samples: int, dim: int) -> np.ndarray:
    """Uniformly sample real vectors in [-1,1]."""
    return np.random.uniform(-1.0, 1.0, size=(samples, dim)).astype(np.float32)

def generate_quantum_superposition(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>."""
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

# ----------------------------------------------------------------------
# Dataset classes
# ----------------------------------------------------------------------
class HybridDataset(Dataset):
    """Dataset that yields either classical or quantum samples."""
    def __init__(self, samples: int, classical_dim: int, quantum_wires: int):
        self.classical = generate_classical_data(samples, classical_dim)
        self.quantum, self.q_labels = generate_quantum_superposition(quantum_wires, samples)

    def __len__(self) -> int:
        return len(self.classical)

    def __getitem__(self, idx: int) -> dict:
        return {
            "classical": torch.tensor(self.classical[idx], dtype=torch.float32),
            "quantum": torch.tensor(self.quantum[idx], dtype=torch.cfloat),
            "q_label": torch.tensor(self.q_labels[idx], dtype=torch.float32),
        }

# ----------------------------------------------------------------------
# Model configuration
# ----------------------------------------------------------------------
@dataclass
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    classical_input_dim: int
    quantum_input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

# ----------------------------------------------------------------------
# Classical autoencoder
# ----------------------------------------------------------------------
class ClassicalAutoencoder(nn.Module):
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        enc_layers = []
        in_dim = config.classical_input_dim
        for h in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, config.classical_input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# ----------------------------------------------------------------------
# Hybrid autoencoder – shares latent space
# ----------------------------------------------------------------------
class HybridAutoencoder(nn.Module):
    """
    Combines a classical encoder/decoder with a quantum encoder that
    operates on the same latent vector. The quantum encoder is implemented
    using a simple parameterised circuit (RealAmplitudes) and a swap‑test
    decoder.  The latent vector is treated as a state vector of size
    2**n_wires, where n_wires = ceil(log2(latent_dim)).
    """
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.classical = ClassicalAutoencoder(config)
        # Quantum side: we will use a simple variational circuit defined in qml module
        # but here we keep a placeholder for the latent dimensionality.
        self.latent_dim = config.latent_dim
        self.n_qubits = int(np.ceil(np.log2(self.latent_dim)))
        # The quantum encoder will be a separate QML module; we expose a method
        # to set it externally.
        self.quantum_encoder = None  # type: ignore[assignment]
        self.quantum_decoder = None  # type: ignore[assignment]

    def set_quantum_encoder(self, encoder):
        """Inject a quantum encoder (from qml module)."""
        self.quantum_encoder = encoder

    def set_quantum_decoder(self, decoder):
        """Inject a quantum decoder (from qml module)."""
        self.quantum_decoder = decoder

    def encode(self, x: torch.Tensor, quantum_state: torch.Tensor) -> torch.Tensor:
        """Encode classical and quantum data into a shared latent vector."""
        z_cls = self.classical.encode(x)
        # quantum encoder expects a state vector; we flatten it into a real vector
        # of size latent_dim.  For simplicity we use the real part only.
        z_q = self.quantum_encoder(quantum_state)
        # Element‑wise fusion (e.g., addition) – could be replaced by a learned fusion layer
        return z_cls + z_q

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to classical space."""
        return self.classical.decode(z)

    def forward(self, x: torch.Tensor, quantum_state: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, quantum_state)
        return self.decode(z)

# ----------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------
def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    dataset: Dataset,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> List[float]:
    """
    Train the hybrid autoencoder on a mixed dataset.
    The loss is a weighted sum of reconstruction error for classical data
    and mean‑squared error between the quantum encoder output and a target
    (here we use the quantum labels from the dataset).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            x = batch["classical"].to(device)
            q = batch["quantum"].to(device)
            q_lbl = batch["q_label"].to(device)

            opt.zero_grad(set_to_none=True)
            recon = model(x, q)
            loss_cls = mse(recon, x)

            # quantum side loss – we compare the latent vector produced by the quantum encoder
            # to the provided quantum label (treated as a scalar target)
            z_q = model.quantum_encoder(q)
            loss_q = mse(z_q.squeeze(-1), q_lbl)

            loss = loss_cls + loss_q
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "HybridDataset",
    "train_hybrid_autoencoder",
]
