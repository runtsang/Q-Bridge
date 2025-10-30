"""Hybrid classical auto‑encoder with quantum latent regularization.

This module implements a `HybridAutoencoder` class that combines a
classical dense encoder/decoder with a quantum variational circuit
used as a regularizer on the latent space.  The class is fully
compatible with PyTorch and can be trained using a simple training
loop that jointly optimises classical weights and quantum parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import the quantum helper
# The quantum module is expected to be in the same package
# and expose a function `get_quantum_circuit`.
try:
    from.quantum_autoencoder import get_quantum_circuit
except Exception:
    # Fallback for when the quantum module is not yet available.
    get_quantum_circuit = None

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # Target distribution for KL divergence
    target_distribution: str = "normal"  # default: normal (N(0,1))
    # Quantum circuit hyper‑parameters
    qparams: dict | None = None  # e.g. {"reps": 5, "basis": "ry"}


# --------------------------------------------------------------------------- #
# Classical network
# --------------------------------------------------------------------------- #
class HybridAutoencoder(nn.Module):
    """Hybrid classical‑quantum auto‑encoder.

    The encoder and decoder are fully‑connected MLPs.  A quantum circuit
    (returned by :func:`get_quantum_circuit`) is applied to the latent
    vector and its expectation value is used as a regularisation term.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
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

        # Quantum circuit
        if get_quantum_circuit is not None:
            self.qc, self.sampler = get_quantum_circuit(
                num_qubits=config.latent_dim,
                **(config.qparams or {})
            )
        else:
            self.qc = None
            self.sampler = None

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

    # --------------------------------------------------------------------- #
    # Quantum penalty
    # --------------------------------------------------------------------- #
    def quantum_penalty(self, latents: torch.Tensor) -> torch.Tensor:
        """Compute a penalty from the quantum circuit.

        The latent vector is used as parameters for the variational
        circuit.  The expectation value of Pauli‑Z on the first qubit is
        returned as a scalar penalty.  The penalty is zero‑mean and
        encourages the latent distribution to match the target.
        """
        if self.sampler is None or self.qc is None:
            return torch.tensor(0.0, device=latents.device, dtype=latents.dtype)

        # Convert latent vector to numpy (requires CPU)
        params = latents.detach().cpu().numpy()
        # The sampler expects a 2‑D array of shape (batch, params)
        result = self.sampler.run(self.qc, parameter_values=params).result()
        statevec = result.get_statevector()
        # Compute expectation of Z on qubit 0 (placeholder)
        exp_z = np.real(statevec[:, 0])  # simplified extraction
        penalty = torch.tensor(exp_z, device=latents.device, dtype=latents.dtype).mean()
        return penalty


# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #
def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    lambda_q: float = 0.1,
    kl_weight: float = 0.01,
) -> list[float]:
    """Simple reconstruction training loop with quantum regularisation.

    Parameters
    ----------
    model
        The :class:`HybridAutoencoder` to train.
    data
        Input data as a 2‑D array (N, D).
    epochs
        Number of epochs.
    batch_size
        Batch size.
    lr
        Learning rate.
    weight_decay
        Weight decay for Adam.
    device
        Target device.
    lambda_q
        Weight of the quantum penalty term.
    kl_weight
        Weight of the KL‑divergence term.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            recon_loss = loss_fn(reconstruction, batch)

            # KL divergence with target normal distribution
            mu = model.encode(batch)
            kl_loss = -0.5 * torch.mean(1 + torch.log(mu.pow(2) + 1e-8) - mu.pow(2))

            # Quantum penalty
            q_penalty = model.quantum_penalty(mu)

            loss = recon_loss + kl_weight * kl_loss + lambda_q * q_penalty
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "HybridAutoencoder",
    "AutoencoderConfig",
    "train_hybrid_autoencoder",
]
