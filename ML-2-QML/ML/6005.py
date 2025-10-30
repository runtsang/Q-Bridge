import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Optional

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
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_vae: bool = False

class VariationalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * latent_dim),
        )
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_logvar = self.net(x)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, hidden: int):
        super().__init__()
        layers = [
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
        ]
        layers.append(nn.Linear(hidden, out_dim))
        self.decoder = nn.Sequential(*layers)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        if config.use_vae:
            self.encoder = VariationalEncoder(
                in_dim=config.input_dim,
                hidden=config.hidden_dims[0],
                latent_dim=config.latent_dim,
            )
        else:
            encoder_layers = []
            in_dim = config.input_dim
            for h in config.hidden_dims:
                encoder_layers.append(nn.Linear(in_dim, h))
                encoder_layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    encoder_layers.append(nn.Dropout(config.dropout))
                in_dim = h
            encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.encoder, VariationalEncoder):
            mu, logvar = self.encoder(x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else:
            return self.encoder(x)
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.encoder, VariationalEncoder):
            z, mu, logvar = self.encode(x)
            recon = self.decode(z)
            return recon, mu, logvar
        else:
            z = self.encode(x)
            return self.decode(z)

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    use_vae: bool = False,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_vae=use_vae,
    )
    return AutoencoderNet(config)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()
    history: List[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if isinstance(model.encoder, VariationalEncoder):
                recon, mu, logvar = model(batch)
                recon_loss = mse_loss(recon, batch)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_div / batch.size(0)
            else:
                recon = model(batch)
                loss = mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
    "VariationalEncoder",
    "Decoder",
]
