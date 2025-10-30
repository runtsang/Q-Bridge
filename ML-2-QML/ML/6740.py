import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List

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
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    residual: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

class AutoencoderNet(nn.Module):
    """
    Fully‑connected autoencoder with optional residual connections and configurable depth.
    """
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent space."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

    def get_latent(self, data: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper to obtain latent codes for a batch."""
        with torch.no_grad():
            return self.encode(data)

    def compute_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean‑squared error, the default reconstruction loss."""
        return nn.functional.mse_loss(recon, target, reduction="mean")

    def evaluate(self, dataloader: DataLoader) -> float:
        """Return average loss over the provided dataloader."""
        self.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(next(self.parameters()).device)
                recon = self(batch)
                loss = self.compute_loss(recon, batch)
                total_loss += loss.item() * batch.size(0)
                total_samples += batch.size(0)
        return total_loss / total_samples

    def save(self, path: str) -> None:
        """Persist the model state dict."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, config: AutoencoderConfig) -> "AutoencoderNet":
        """Instantiate and load a saved model."""
        model = cls(config)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    residual: bool = False,
) -> AutoencoderNet:
    """
    Factory that mirrors the quantum helper, returning a configured network.
    """
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        residual=residual,
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
    """
    Simple reconstruction training loop with early stopping.
    Returns a list of epoch‑level training losses.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[float] = []

    best_loss = float("inf")
    patience = model.config.early_stopping_patience
    counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = model.compute_loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        # Early‑stopping check
        if epoch_loss < best_loss - model.config.early_stopping_min_delta:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    return history

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
]
