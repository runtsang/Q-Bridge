import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Optional

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
    """Configuration for a flexible auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_layer_norm: bool = False
    skip_connections: bool = False

class AutoencoderNet(nn.Module):
    """A versatile fully‑connected auto‑encoder with optional layer‑norm and skip‑connections."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential()
        in_dim = cfg.input_dim
        for i, h in enumerate(cfg.hidden_dims):
            self.encoder.add_module(f"enc_lin_{i}", nn.Linear(in_dim, h))
            if cfg.use_layer_norm:
                self.encoder.add_module(f"enc_ln_{i}", nn.LayerNorm(h))
            self.encoder.add_module(f"enc_relu_{i}", nn.ReLU())
            if cfg.dropout > 0.0:
                self.encoder.add_module(f"enc_drop_{i}", nn.Dropout(cfg.dropout))
            in_dim = h
        self.encoder.add_module("enc_latent", nn.Linear(in_dim, cfg.latent_dim))

        self.decoder = nn.Sequential()
        in_dim = cfg.latent_dim
        for i, h in enumerate(reversed(cfg.hidden_dims)):
            self.decoder.add_module(f"dec_lin_{i}", nn.Linear(in_dim, h))
            if cfg.use_layer_norm:
                self.decoder.add_module(f"dec_ln_{i}", nn.LayerNorm(h))
            self.decoder.add_module(f"dec_relu_{i}", nn.ReLU())
            if cfg.dropout > 0.0:
                self.decoder.add_module(f"dec_drop_{i}", nn.Dropout(cfg.dropout))
            in_dim = h
        self.decoder.add_module("dec_out", nn.Linear(in_dim, cfg.input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        if self.cfg.skip_connections:
            if z.shape[-1] == x.shape[-1]:
                x = x + z
        return self.decode(z)

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    use_layer_norm: bool = False,
    skip_connections: bool = False,
) -> AutoencoderNet:
    """Factory that returns a configured classical auto‑encoder."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
        skip_connections=skip_connections,
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
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> List[float]:
    """Standard reconstruction training loop with optional progress logging."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
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
        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs:3d} – loss: {epoch_loss:.6f}")
    return history

__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
