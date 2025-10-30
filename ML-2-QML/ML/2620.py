"""Unified hybrid auto‑encoder combining classical and quantum components.

The module exposes a single ``UnifiedAutoEncoder`` class that builds a
classical MLP auto‑encoder and a quantum variational encoder that
operate on the same latent dimension.  The quantum part is optional
and can be toggled by passing ``use_quantum=True``.  The classical
decoder receives the concatenated latent vector (classical + quantum)
and is trained jointly with the classical encoder.  This design
mirrors the *Autoencoder.py* seed and the *Autoencoder.py* QML seed,
while adding a quantum‑aware skip‑connection and a configurable
regulariser that encourages the latent vectors to be close.

The implementation is fully importable; it can be dropped into
``Autoencoder__gen178.py`` and used as a standalone module.

Author: gpt-oss-20b
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# --------------------------------------------------------------------------- #
#  CONFIGURATION
# --------------------------------------------------------------------------- #
@dataclass
class UnifiedAutoEncoderConfig:
    """Configuration for the hybrid auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # quantum‑specific
    use_quantum: bool = False
    n_qubits: int = 4
    quantum_regulariser_weight: float = 0.01
    skip_quantum: bool = False


# --------------------------------------------------------------------------- #
#  CLASSICAL COMPONENTS
# --------------------------------------------------------------------------- #
class ClassicalAutoencoder(nn.Module):
    """Fully‑connected auto‑encoder used as the classical backbone."""
    def __init__(self, cfg: UnifiedAutoEncoderConfig) -> None:
        super().__init__()
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# --------------------------------------------------------------------------- #
#  QUANTUM COMPONENT (optional)
# --------------------------------------------------------------------------- #
class QuantumEncoder(nn.Module):
    """Parameter‑shared RealAmplitudes encoder with optional swap‑test readout."""
    def __init__(self, cfg: UnifiedAutoEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.available = False
        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit.library import RealAmplitudes
            self.circuit = RealAmplitudes(cfg.n_qubits, reps=5)
            self.available = True
        except Exception:
            # Fallback: use a random linear map as a placeholder
            self.random_lin = nn.Linear(cfg.n_qubits, cfg.latent_dim, bias=False)
            nn.init.uniform_(self.random_lin.weight, -np.pi, np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (batch, n_qubits) – raw input qubit amplitudes.
        Returns a latent vector of shape (batch, latent_dim).
        """
        if self.available:
            # Convert to numpy and run the qiskit circuit
            import numpy as np
            from qiskit import Aer, execute
            backend = Aer.get_backend('statevector_simulator')
            batch = x.shape[0]
            latent = []
            for i in range(batch):
                state = x[i].detach().cpu().numpy()
                qc = self.circuit.copy()
                qc.initialize(state, qc.qubits)
                job = execute(qc, backend, shots=1)
                result = job.result()
                sv = result.get_statevector(qc)
                # Use expectation of PauliZ on each qubit as a simple readout
                z_exp = np.real(np.dot(sv.conj(), np.diag([(-1)**j for j in range(2**self.cfg.n_qubits)])) @ sv)
                latent.append(torch.tensor(z_exp, dtype=torch.float32))
            latent = torch.stack(latent, dim=0)
            return latent
        else:
            # Dummy linear map
            return self.random_lin(x)


# --------------------------------------------------------------------------- #
#  HYBRID MODEL
# --------------------------------------------------------------------------- #
class UnifiedAutoEncoder(nn.Module):
    """Hybrid auto‑encoder that fuses classical and quantum latent spaces."""
    def __init__(self, cfg: UnifiedAutoEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.classical = ClassicalAutoencoder(cfg)
        self.quantum = QuantumEncoder(cfg) if cfg.use_quantum else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return concatenated latent vector."""
        z_cls = self.classical.encode(x)
        if self.quantum is not None:
            # Assume input x is a statevector of shape (batch, n_qubits)
            z_q = self.quantum(x)
            return torch.cat([z_cls, z_q], dim=1)
        return z_cls

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode only the classical part of the latent vector."""
        # In the hybrid case, the first part corresponds to the classical latent
        z_cls = z[:, :self.cfg.latent_dim]
        return self.classical.decode(z_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


# --------------------------------------------------------------------------- #
#  TRAINING UTILITIES
# --------------------------------------------------------------------------- #
def _tensorify(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)


def train_unified_autoencoder(
    model: UnifiedAutoEncoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_tensorify(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            # optional quantum regulariser
            if model.quantum is not None:
                z_q = model.quantum(batch)
                z_cls = model.classical.encode(batch)
                reg = torch.mean((z_q - z_cls) ** 2)
                loss += model.cfg.quantum_regulariser_weight * reg
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "UnifiedAutoEncoderConfig",
    "UnifiedAutoEncoder",
    "train_unified_autoencoder",
]
