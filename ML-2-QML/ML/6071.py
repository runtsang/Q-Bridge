import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Optional

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Coerce data to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        return data
    tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderConfig:
    """Configuration for the auto‑encoder network."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1

class Autoencoder__gen491(nn.Module):
    """Fully‑connected auto‑encoder with early stopping and hybrid loss."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Encoder
        enc_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 1e-4,
        device: Optional[torch.device] = None,
        val_split: float = 0.1,
    ) -> List[float]:
        """Train the auto‑encoder with early stopping.

        Args:
            data: Input data tensor of shape (N, input_dim).
            epochs: Maximum number of epochs.
            batch_size: Batch size.
            lr: Learning rate.
            weight_decay: Weight decay for Adam.
            early_stopping_patience: Number of epochs with no improvement after which training will be stopped.
            early_stopping_delta: Minimum change to qualify as improvement.
            device: Device to run training on.
            val_split: Fraction of data to use for validation.

        Returns:
            A list of validation losses per epoch.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        data = _as_tensor(data).to(device)

        # Split data
        n = len(data)
        idx = torch.randperm(n)
        split = int(n * (1 - val_split))
        train_idx, val_idx = idx[:split], idx[split:]
        train_loader = DataLoader(
            TensorDataset(data[train_idx]), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(data[val_idx]), batch_size=batch_size, shuffle=False
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        val_losses: List[float] = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for batch, in train_loader:
                optimizer.zero_grad(set_to_none=True)
                recon = self(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(train_loader.dataset)

            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch, in val_loader:
                    recon = self(batch)
                    loss = loss_fn(recon, batch)
                    val_loss += loss.item() * batch.size(0)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss - early_stopping_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

        return val_losses

    def predict(self, inputs: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return reconstructions for the given inputs."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        with torch.no_grad():
            return self.forward(inputs.to(device)).cpu()

def Autoencoder__gen491_factory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
) -> Autoencoder__gen491:
    """Convenience factory mirroring the original API."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return Autoencoder__gen491(cfg)

__all__ = [
    "Autoencoder__gen491",
    "Autoencoder__gen491_factory",
    "AutoencoderConfig",
]
