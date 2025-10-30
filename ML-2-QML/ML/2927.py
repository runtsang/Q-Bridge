import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class HybridQuantumAutoencoder(nn.Module):
    """A hybrid autoencoder that combines a CNN feature extractor, a classical MLP encoder,
    and a classical MLP decoder. It is inspired by the QuantumNAT CNN + the
    fully‑connected autoencoder from the second reference."""
    def __init__(self, config: AutoencoderConfig, conv_out_dim: int = 16 * 7 * 7) -> None:
        super().__init__()
        # CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Linear projection from conv features
        self.fc_proj = nn.Sequential(
            nn.Linear(conv_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, config.latent_dim),
        )
        # Classical MLP encoder (from AutoencoderNet)
        encoder_layers = []
        in_dim = config.latent_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.mlp_encoder = nn.Sequential(*encoder_layers)

        # Classical MLP decoder (from AutoencoderNet)
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.mlp_decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        flat = features.view(features.shape[0], -1)
        latent = self.fc_proj(flat)
        return self.mlp_encoder(latent)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp_decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def train_autoencoder(
    model: nn.Module,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop identical to the original, but accepts any nn.Module."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
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

__all__ = ["HybridQuantumAutoencoder", "AutoencoderConfig", "train_autoencoder"]
