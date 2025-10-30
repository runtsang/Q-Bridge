import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Iterable

class QuanvolutionFilter(nn.Module):
    """Patch‑wise 2x2 convolution that mimics a simple quantum kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_quanvolution: bool = False

class AutoencoderHybrid(nn.Module):
    """Classical dense + optional quanvolution autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_extractor = QuanvolutionFilter() if config.use_quanvolution else nn.Identity()

        # Determine feature dimensionality after optional quanvolution
        dummy = torch.zeros(1, 1, 28, 28)
        feat_dim = self.feature_extractor(dummy).shape[1] if config.use_quanvolution else 28 * 28

        # Encoder
        enc_layers = []
        in_dim = feat_dim
        for h in config.hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(config.dropout)]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(config.dropout)]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, feat_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.feature_extractor(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

def train_autoencoder(
    model: AutoencoderHybrid,
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

__all__ = ["AutoencoderHybrid", "AutoencoderConfig", "train_autoencoder"]
