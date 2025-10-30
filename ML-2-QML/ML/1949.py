"""Enhanced PyTorch autoencoder with denoising, early stopping, and flexible loss."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable, Callable, List, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    activation: Callable = nn.ReLU

class Autoencoder(nn.Module):
    """A fully‑connected autoencoder with optional denoising and early stopping."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_mlp(
            config.input_dim,
            config.hidden_dims,
            config.latent_dim,
            config.dropout,
            config.activation,
            encode=True
        )
        self.decoder = self._build_mlp(
            config.latent_dim,
            config.hidden_dims[::-1],
            config.input_dim,
            config.dropout,
            config.activation,
            encode=False
        )

    def _build_mlp(self,
                   in_dim: int,
                   hidden_dims: Tuple[int,...],
                   out_dim: int,
                   dropout: float,
                   activation: Callable,
                   encode: bool) -> nn.Sequential:
        layers: List[nn.Module] = []
        curr_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h
        layers.append(nn.Linear(curr_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

    def train_autoencoder(self,
                          data: torch.Tensor,
                          *,
                          epochs: int = 100,
                          batch_size: int = 64,
                          lr: float = 1e-3,
                          weight_decay: float = 0.0,
                          loss_fn: nn.Module = nn.MSELoss(),
                          early_stopping: int = 10,
                          noise_std: float = 0.0,
                          device: Optional[torch.device] = None) -> List[float]:
        """Train the autoencoder with optional denoising and early stopping.

        Parameters
        ----------
        data : torch.Tensor
            Training data, shape (N, input_dim).
        epochs : int
            Number of training epochs.
        batch_size : int
            Size of mini‑batches.
        lr : float
            Learning rate.
        weight_decay : float
            L2 regularisation.
        loss_fn : nn.Module
            Loss function to optimise.
        early_stopping : int
            Number of epochs with no improvement after which training stops.
        noise_std : float
            Standard deviation of Gaussian noise added to inputs for denoising.
        device : torch.device | None
            Device to run training on; defaults to CUDA if available.

        Returns
        -------
        List[float]
            Training loss per epoch.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        history: List[float] = []

        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            self.train()
            for (batch,) in loader:
                batch = batch.to(device)
                if noise_std > 0.0:
                    batch_noisy = batch + torch.randn_like(batch) * noise_std
                else:
                    batch_noisy = batch
                optimizer.zero_grad(set_to_none=True)
                recon = self(batch_noisy)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)

            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if early_stopping > 0 and epochs_no_improve >= early_stopping:
                print(f"Early stopping at epoch {epoch+1}")
                break
        return history

    def evaluate(self, data: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return reconstruction errors for the provided data."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        with torch.no_grad():
            recon = self.forward(_as_tensor(data).to(device))
            return torch.mean((recon - _as_tensor(data).to(device)) ** 2, dim=1)

__all__ = ["AutoencoderConfig", "Autoencoder"]
