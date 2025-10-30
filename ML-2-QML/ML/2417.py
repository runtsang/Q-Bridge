from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qml import QuantumAutoencoderCircuit, HybridFunction

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
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
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
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

class HybridAutoencoder(nn.Module):
    """Hybrid classical‑quantum autoencoder that concatenates a classical latent vector
    with a quantum‑encoded representation. The quantum part is implemented as a
    parameterised variational circuit whose expectation values are differentiable
    via a custom autograd function."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        qnum_qubits: int = 32,
        qansatz_reps: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )
        self.quantum_encoder = QuantumAutoencoderCircuit(
            num_qubits=qnum_qubits,
            reps=qansatz_reps,
        )
        # Decoder receives the concatenated classical+quantum vector
        decoder_config = AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim + qnum_qubits,
            hidden_dims=hidden_dims[::-1],
            dropout=dropout,
        )
        self.decoder = AutoencoderNet(decoder_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        q_out = HybridFunction.apply(latent, self.quantum_encoder)
        combined = torch.cat([latent, q_out], dim=-1)
        return self.decoder(combined)

def train_hybrid_autoencoder(
    model: HybridAutoencoder,
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["AutoencoderConfig", "AutoencoderNet", "HybridAutoencoder", "train_hybrid_autoencoder"]
