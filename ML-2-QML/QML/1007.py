import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Iterable, List

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 3
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    kl_weight: float = 1e-3

class QuantumEncoder(nn.Module):
    """Quantum encoder that outputs mean and log‑variance for a VAE."""
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_qubits = latent_dim * 2  # first half: mu, second half: logvar
        self.params = nn.Parameter(torch.randn(self.num_qubits))
        self.dev = qml.device("default.qubit.autograd", wires=self.num_qubits)
        self.qnode = qml.qnode(self._circuit, device=self.dev, interface="torch")

    def _circuit(self, inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Encode input features via RX rotations
        for i in range(self.num_qubits):
            idx = i % self.input_dim
            qml.RX(inputs[idx], wires=i)
        qml.templates.RealAmplitudes(params, wires=range(self.num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pad or truncate input to match number of qubits
        if inputs.shape[-1] < self.num_qubits:
            pad = torch.zeros(self.num_qubits - inputs.shape[-1], device=inputs.device)
            inputs = torch.cat([inputs, pad], dim=-1)
        else:
            inputs = inputs[..., :self.num_qubits]
        out = self.qnode(inputs, self.params)
        mu = out[:self.latent_dim]
        logvar = out[self.latent_dim:]
        return mu, logvar

class AutoencoderGen302(nn.Module):
    """Hybrid quantum‑classical VAE autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = QuantumEncoder(config.input_dim, config.latent_dim)
        # Decoder (classical MLP)
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
        self.kl_weight = config.kl_weight

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def AutoencoderGen302_factory(
    input_dim: int,
    *,
    latent_dim: int = 3,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    kl_weight: float = 1e-3,
) -> AutoencoderGen302:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        kl_weight=kl_weight,
    )
    return AutoencoderGen302(config)

def train_autoencoder(
    model: AutoencoderGen302,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss(reduction="sum")
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            recon_loss = loss_fn(recon, batch)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_loss = torch.mean(kl)
            loss = recon_loss + model.kl_weight * kl_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["AutoencoderGen302", "AutoencoderGen302_factory", "train_autoencoder"]
