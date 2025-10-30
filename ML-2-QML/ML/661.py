"""AutoencoderGen318: Classical autoencoder with optional quantum encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

import pennylane as qml

__all__ = [
    "AutoencoderGen318",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
    "train_hybrid",
    "cluster_latent",
]


@dataclass
class AutoencoderConfig:
    """Configuration for the autoencoder."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # Optional: use a quantum encoder instead of the classical MLP
    use_quantum: bool = False
    # Optional: user supplied quantum circuit builder
    q_circuit_builder: Callable[[int, int], nn.Module] | None = None


class QuantumEncoder(nn.Module):
    """Simple quantum encoder that maps an input vector to a latent vector."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dev = qml.device("default.qubit", wires=latent_dim)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            # Encode each feature as an RX rotation
            for i, val in enumerate(x):
                qml.RX(val, wires=i % latent_dim)
            # Entangle all qubits
            qml.templates.BasicEntanglerLayers(
                weights=torch.zeros((1, latent_dim)), wires=range(latent_dim)
            )
            # Return expectation values of PauliZ
            return [qml.expval(qml.PauliZ(w)) for w in range(latent_dim)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        return self.circuit(x)


class AutoencoderNet(nn.Module):
    """Multi‑layer perceptron autoencoder with an optional quantum encoder."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Encoder
        if cfg.use_quantum and cfg.q_circuit_builder is not None:
            self.encoder = cfg.q_circuit_builder(cfg.input_dim, cfg.latent_dim)
        elif cfg.use_quantum:
            self.encoder = QuantumEncoder(cfg.input_dim, cfg.latent_dim)
        else:
            encoder_layers: list[nn.Module] = []
            in_dim = cfg.input_dim
            for h in cfg.hidden_dims:
                encoder_layers.append(nn.Linear(in_dim, h))
                encoder_layers.append(nn.ReLU())
                if cfg.dropout > 0.0:
                    encoder_layers.append(nn.Dropout(cfg.dropout))
                in_dim = h
            encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: list[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def AutoencoderGen318(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_quantum: bool = False,
    q_circuit_builder: Callable[[int, int], nn.Module] | None = None,
) -> AutoencoderNet:
    """Convenience factory mirroring the original API."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_quantum=use_quantum,
        q_circuit_builder=q_circuit_builder,
    )
    return AutoencoderNet(cfg)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Classic reconstruction training loop."""
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


def train_hybrid(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Hybrid training loop that supports quantum encoders."""
    return train_autoencoder(
        model, data, epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay, device=device
    )


def cluster_latent(
    model: AutoencoderNet,
    data: torch.Tensor,
    n_clusters: int = 2,
) -> KMeans:
    """Cluster the latent representations produced by the encoder."""
    latents = model.encode(_as_tensor(data).to(next(model.parameters()).device))
    latents_np = latents.detach().cpu().numpy()
    return KMeans(n_clusters=n_clusters).fit(latents_np)


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor
