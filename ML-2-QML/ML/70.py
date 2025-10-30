import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List

@dataclass
class AutoencoderConfig:
    """Configuration for a flexible MLP autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    batch_norm: bool = False
    activation: nn.Module = nn.ReLU()

class Autoencoder(nn.Module):
    """A fully‑connected autoencoder with optional batch‑norm and dropout."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        self.encoder = self._make_mlp(
            in_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.latent_dim,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            activation=config.activation,
        )
        self.decoder = self._make_mlp(
            in_dim=config.latent_dim,
            hidden_dims=config.hidden_dims[::-1],
            out_dim=config.input_dim,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            activation=config.activation,
        )

    @staticmethod
    def _make_mlp(in_dim: int,
                  hidden_dims: Tuple[int,...],
                  out_dim: int,
                  dropout: float,
                  batch_norm: bool,
                  activation: nn.Module) -> nn.Sequential:
        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

    def encode_batch(self, data: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=False)
        z_list = []
        with torch.no_grad():
            for batch, in loader:
                z_list.append(self.encode(batch.to(self.device)))
        return torch.cat(z_list, dim=0)

    def decode_batch(self, z: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        loader = DataLoader(TensorDataset(z), batch_size=batch_size, shuffle=False)
        x_list = []
        with torch.no_grad():
            for batch, in loader:
                x_list.append(self.decode(batch.to(self.device)))
        return torch.cat(x_list, dim=0)

    def reconstruction_loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(recon, x, reduction="mean")

    def train_autoencoder(
        self,
        data: torch.Tensor,
        epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
        early_stop_patience: int | None = None,
    ) -> List[float]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon = self(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)

            if early_stop_patience is not None:
                if epoch_loss < best_loss - 1e-4:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    break
        return history

    def evaluate(self, data: torch.Tensor, device: torch.device | None = None) -> float:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        with torch.no_grad():
            recon = self.forward(data.to(device))
            loss = self.reconstruction_loss(data.to(device), recon).item()
        return loss

def create_autoencoder(input_dim: int,
                       latent_dim: int = 32,
                       hidden_dims: Tuple[int,...] = (128, 64),
                       dropout: float = 0.1,
                       batch_norm: bool = False,
                       activation: nn.Module = nn.ReLU()) -> Autoencoder:
    """Factory that returns a configured :class:`Autoencoder`."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        batch_norm=batch_norm,
        activation=activation,
    )
    return Autoencoder(config)

__all__ = ["Autoencoder", "AutoencoderConfig", "create_autoencoder"]
