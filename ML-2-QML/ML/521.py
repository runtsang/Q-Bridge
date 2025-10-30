import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    """Configuration for a hierarchical, residual autoencoder."""
    input_dim: int
    latent_dims: Tuple[int,...] = (32,)
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    residual: bool = False

class AutoencoderNet(nn.Module):
    """A multi‑layer perceptron autoencoder with optional skip‑connections."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Encoder
        enc_layers = []
        in_d = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_d, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_d = h
        # Residual block
        if cfg.residual:
            enc_layers.append(nn.Linear(in_d, in_d))
            enc_layers.append(nn.ReLU())
        # Latent stack
        self.latent_stacks = nn.ModuleList()
        for l in cfg.latent_dims:
            self.latent_stacks.append(nn.Linear(in_d, l))
        enc_layers.append(nn.Sequential(*self.latent_stacks))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_d = cfg.latent_dims[-1]
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_d, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_d = h
        dec_layers.append(nn.Linear(in_d, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(
    input_dim: int,
    *,
    latent_dims: Tuple[int,...] = (32,),
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    residual: bool = False,
) -> AutoencoderNet:
    """Factory that returns a configured hierarchical autoencoder."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dims=latent_dims,
        hidden_dims=hidden_dims,
        dropout=dropout,
        residual=residual,
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
    """Train the model and return a list of training losses."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(_as_tensor(data)),
        batch_size=batch_size,
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        history.append(epoch_loss / len(loader.dataset))
    return history

def _as_tensor(data):
    """Utility to cast data to float32 tensor."""
    if isinstance(data, torch.Tensor):
        return data.to(dtype=torch.float32)
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
