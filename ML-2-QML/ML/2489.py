import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Iterable

@dataclass
class AutoencoderConfig:
    input_channels: int = 1
    input_size: int = 28
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class HybridQuantumAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        # CNN encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(config.input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Determine flattened feature size
        dummy = torch.zeros(1, config.input_channels, config.input_size, config.input_size)
        with torch.no_grad():
            flat_size = self.encoder_cnn(dummy).view(1, -1).size(1)
        # Latent fully connected
        self.fc_latent = nn.Sequential(
            nn.Linear(flat_size, config.latent_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        # Decoder fully connected
        self.decoder_fc = nn.Sequential(
            nn.Linear(config.latent_dim, flat_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        # Decoder CNN to reconstruct image
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, config.input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.output_shape = (config.input_channels, config.input_size, config.input_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder_cnn(x)
        flat = features.view(x.size(0), -1)
        latent = self.fc_latent(flat)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        flat = self.decoder_fc(latent)
        batch = latent.size(0)
        h = w = int(self.output_shape[1] // 4)
        flat = flat.view(batch, 16, h, w)
        out = self.decoder_cnn(flat)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon

def HybridQuantumAutoencoderFactory(
    input_channels: int = 1,
    input_size: int = 28,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridQuantumAutoencoder:
    config = AutoencoderConfig(
        input_channels=input_channels,
        input_size=input_size,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridQuantumAutoencoder(config)

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

def train_autoencoder(
    model: HybridQuantumAutoencoder,
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
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridQuantumAutoencoder", "HybridQuantumAutoencoderFactory", "train_autoencoder", "AutoencoderConfig"]
