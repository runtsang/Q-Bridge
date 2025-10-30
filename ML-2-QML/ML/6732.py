import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class ClassicalAutoEncoder(nn.Module):
    """Dense‑layer autoencoder serving as the classical backbone."""
    def __init__(self, config: AutoencoderConfig):
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class UnifiedAutoencoder(nn.Module):
    """Hybrid classical‑quantum autoencoder that fuses a classical encoder
    with an optional quantum latent encoder.  The latent vector is
    concatenated before decoding, allowing end‑to‑end training.
    """
    def __init__(
        self,
        config: AutoencoderConfig,
        quantum_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.classical = ClassicalAutoEncoder(config)
        self.quantum_encoder = quantum_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_class = self.classical.encode(x)
        if self.quantum_encoder is not None:
            z_quant = self.quantum_encoder.encode(x)
            z = torch.cat([z_class, z_quant], dim=-1)
        else:
            z = z_class
        return self.classical.decode(z)

def train_unified(
    model: UnifiedAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """End‑to‑end training loop for the hybrid autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "AutoencoderConfig",
    "ClassicalAutoEncoder",
    "UnifiedAutoencoder",
    "train_unified",
]
