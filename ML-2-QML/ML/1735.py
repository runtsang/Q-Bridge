import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, Optional, List

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert any iterable or tensor to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    """Configuration for Autoencoder__gen320."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_batch_norm: bool = False
    denoise: bool = False
    noise_level: float = 0.05
    early_stopping_patience: int = 10

class Autoencoder__gen320(nn.Module):
    """Hybrid dense autoencoder with optional denoising, residuals, and early‑stopping."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            if config.use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            if config.use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent representation."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""
        latent = self.encode(x)
        recon = self.decode(latent)
        if x.shape == recon.shape:
            recon += x  # simple residual skip
        return recon

def Autoencoder__gen320_factory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_batch_norm: bool = False,
    denoise: bool = False,
    noise_level: float = 0.05,
    early_stopping_patience: int = 10,
) -> Autoencoder__gen320:
    """Return a configured Autoencoder__gen320 instance."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        denoise=denoise,
        noise_level=noise_level,
        early_stopping_patience=early_stopping_patience,
    )
    return Autoencoder__gen320(cfg)

def train_autoencoder__gen320(
    model: Autoencoder__gen320,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Training loop with optional denoising and early stopping."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    best_loss = float("inf")
    patience = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            if model.config.denoise:
                noise = torch.randn_like(batch) * model.config.noise_level
                noisy_batch = batch + noise
            else:
                noisy_batch = batch
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(noisy_batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1
        if patience >= model.config.early_stopping_patience:
            break
    return history

__all__ = [
    "Autoencoder__gen320",
    "Autoencoder__gen320_factory",
    "train_autoencoder__gen320",
    "AutoencoderConfig",
]
