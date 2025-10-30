"""HybridSamplerQNN: Classical wrapper for a quantum sampler network.

The module defines a PyTorch ``nn.Module`` that first compresses the input
through a lightweight auto‑encoder and then forwards the latent vector to a
Qiskit ``SamplerQNN``.  The output is a probability distribution over the
specified output shape, ready for downstream loss functions such as cross‑entropy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# Classical auto‑encoder utilities (adapted from the reference Autoencoder)
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


@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron auto‑encoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
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

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


# --------------------------------------------------------------------------- #
# Hybrid sampler module
# --------------------------------------------------------------------------- #

@dataclass
class HybridConfig:
    """Configuration for the hybrid sampler."""
    input_dim: int = 2
    latent_dim: int = 4
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    output_shape: int = 2


class HybridSamplerQNN(nn.Module):
    """PyTorch wrapper that combines a classical auto‑encoder and a Qiskit SamplerQNN."""

    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        # Classical encoder
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=config.input_dim,
                latent_dim=config.latent_dim,
                hidden_dims=config.hidden_dims,
                dropout=config.dropout,
            )
        )

        # Quantum sampler
        from qiskit.circuit import ParameterVector
        from qiskit import QuantumCircuit
        from qiskit.primitives import StatevectorSampler
        from qiskit_machine_learning.neural_networks import SamplerQNN

        # Circuit with 2 qubits: 2 input parameters + 4 weight parameters
        inputs = ParameterVector("x", 2)
        weights = ParameterVector("w", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
        )
        self.output_shape = config.output_shape

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode the input, sample from the quantum circuit, and return probabilities."""
        # Encode
        latent = self.autoencoder.encode(inputs)

        # Convert latent to NumPy for the quantum sampler
        latent_np = latent.detach().cpu().numpy()

        # Forward through the quantum circuit
        probs = self.qnn.forward(latent_np)

        # Convert back to a Torch tensor
        return torch.from_numpy(probs).float()

# --------------------------------------------------------------------------- #
# Optional training helper (mirrors the reference training loop)
# --------------------------------------------------------------------------- #

def train_hybrid(
    model: HybridSamplerQNN,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple training loop for the hybrid model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            # Forward pass: obtain probabilities
            probs = model(batch)
            # Ground truth: use the first column as class label for demo
            labels = batch[:, 0].long()
            loss = loss_fn(probs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridSamplerQNN",
    "HybridConfig",
    "AutoencoderNet",
    "AutoencoderConfig",
    "train_hybrid",
]
