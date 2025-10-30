"""Extended PyTorch autoencoder with residual connections and early stopping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


class ResidualBlock(nn.Module):
    """Simple residual block with optional dropout."""
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.skip = nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dropout(self.linear(x))) + self.skip(x)


@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    activation: Callable[[], nn.Module] = nn.ReLU
    use_residual: bool = True


class AutoencoderNet(nn.Module):
    """A multilayer perceptron autoencoder with optional residual blocks."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        def make_layers(dims: Tuple[int,...]) -> nn.Sequential:
            layers: list[nn.Module] = []
            in_dim = dims[0]
            for out_dim in dims[1:]:
                if config.use_residual:
                    layers.append(ResidualBlock(in_dim, out_dim, config.activation, config.dropout))
                else:
                    layers.append(nn.Linear(in_dim, out_dim))
                    layers.append(config.activation())
                    if config.dropout > 0.0:
                        layers.append(nn.Dropout(config.dropout))
                in_dim = out_dim
            return nn.Sequential(*layers)

        # Encoder
        encoder_dims = (config.input_dim, *config.hidden_dims, config.latent_dim)
        self.encoder = make_layers(encoder_dims)

        # Decoder
        decoder_dims = (config.latent_dim, *reversed(config.hidden_dims), config.input_dim)
        self.decoder = make_layers(decoder_dims)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    activation: Callable[[], nn.Module] = nn.ReLU,
    use_residual: bool = True,
) -> AutoencoderNet:
    """Factory that returns a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        use_residual=use_residual,
    )
    return AutoencoderNet(config)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stopping: int | None = None,
) -> list[float]:
    """
    Train the autoencoder and return the epoch‑wise MSE loss history.
    If early_stopping is set, training stops when the validation loss does not improve
    for that many consecutive epochs. The function uses a simple train/val split.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(_as_tensor(data))
    num_train = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [num_train, len(dataset) - num_train])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        history.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += loss_fn(recon, batch).item() * batch.size(0)
        val_loss /= len(val_loader.dataset)

        if early_stopping is not None:
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    return history


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
]
