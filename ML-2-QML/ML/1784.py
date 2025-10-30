import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Sequence

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Sequence[int] = (128, 64)
    activation: nn.Module = nn.ReLU()
    dropout: float = 0.1
    early_stopping: bool = False
    patience: int = 10

class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron auto‑encoder with configurable depth."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._make_block(
            in_dim=config.input_dim,
            dims=config.hidden_dims,
            out_dim=config.latent_dim,
            activation=config.activation,
            dropout=config.dropout,
        )
        self.decoder = self._make_block(
            in_dim=config.latent_dim,
            dims=list(reversed(config.hidden_dims)),
            out_dim=config.input_dim,
            activation=config.activation,
            dropout=config.dropout,
        )

    @staticmethod
    def _make_block(
        in_dim: int,
        dims: Sequence[int],
        out_dim: int,
        activation: nn.Module,
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for dim in dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = dim
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, **kwargs) -> AutoencoderNet:
    """Factory mirroring the quantum helper – returns a configured network."""
    cfg = AutoencoderConfig(input_dim=input_dim, **kwargs)
    return AutoencoderNet(cfg)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stopping: bool = False,
    patience: int = 15,
) -> list[float]:
    """Reconstruction training loop with optional early‑stopping."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    best_loss = float("inf")
    best_state = None
    counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        if early_stopping:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if best_state:
                        model.load_state_dict(best_state)
                    break
    return history

__all__ = ["AutoencoderConfig", "AutoencoderNet", "Autoencoder", "train_autoencoder"]
