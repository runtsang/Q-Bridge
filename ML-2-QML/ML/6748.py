import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Callable

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
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    kl_weight: float = 0.0
    patience: int = 10

class ResidualDense(nn.Module):
    """A lightweight residual dense block used in the encoder."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
        self.res = nn.Linear(in_dim, out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.res(x)

class AutoencoderNet(nn.Module):
    """Hybrid encoder–decoder with a quantum‑enabled decoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        encoder_layers: List[Callable] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(ResidualDense(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder will be attached later via set_decoder.
        self.decoder = None

    def set_decoder(self, decoder: nn.Module) -> None:
        """Attach a decoder after construction."""
        self.decoder = decoder

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("Decoder not set. Call set_decoder() before forward.")
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        self.latents = latent
        return self.decode(latent)

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    kl_weight: float = 0.0,
    patience: int = 10,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        kl_weight=kl_weight,
        patience=patience,
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
    device: torch.device | None = None,
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
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)

            # KL‑like latent regularizer (L2 norm of latent mean)
            if hasattr(model, "latents"):
                latent_mean = model.latents.mean(dim=1)
                loss += model.config.kl_weight * latent_mean.pow(2).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        # Early‑stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= model.config.patience:
                break

    return history

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "ResidualDense",
    "train_autoencoder",
]
