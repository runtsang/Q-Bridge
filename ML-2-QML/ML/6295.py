import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Optional

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "VariationalAutoencoder",
    "train_autoencoder",
    "train_vae",
]

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
    """Configuration values for :class:`AutoencoderNet` and :class:`VariationalAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    batchnorm: bool = False
    early_stopping_patience: Optional[int] = None


class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder with optional batch‑norm."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._build_mlp(
            config.input_dim, config.hidden_dims, config.latent_dim, config.batchnorm
        )
        self.decoder = self._build_mlp(
            config.latent_dim,
            tuple(reversed(config.hidden_dims)),
            config.input_dim,
            config.batchnorm,
        )

    @staticmethod
    def _build_mlp(
        in_dim: int,
        hidden: Tuple[int,...],
        out_dim: int,
        batchnorm: bool,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.05))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


class VariationalAutoencoder(nn.Module):
    """A variational auto‑encoder with a simple MLP encoder/decoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._build_mlp(
            config.input_dim, config.hidden_dims, config.latent_dim * 2, config.batchnorm
        )
        self.decoder = self._build_mlp(
            config.latent_dim,
            tuple(reversed(config.hidden_dims)),
            config.input_dim,
            config.batchnorm,
        )

    @staticmethod
    def _build_mlp(
        in_dim: int,
        hidden: Tuple[int,...],
        out_dim: int,
        batchnorm: bool,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.05))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    early_stopping_patience: Optional[int] = None,
) -> List[float]:
    """Simple reconstruction training loop returning the loss history."""
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
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        if early_stopping_patience is not None:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    break
    return history


def train_vae(
    model: VariationalAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    early_stopping_patience: Optional[int] = None,
) -> List[float]:
    """Training loop for the VAE returning the total loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[float] = []
    best_loss = float("inf")
    patience = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch)
            recon_loss = nn.functional.mse_loss(recon, batch, reduction="sum")
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon_loss + kl_loss) / batch.size(0)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        if early_stopping_patience is not None:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    break
    return history


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    batchnorm: bool = False,
    early_stopping_patience: Optional[int] = None,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        batchnorm=batchnorm,
        early_stopping_patience=early_stopping_patience,
    )
    return AutoencoderNet(config)
