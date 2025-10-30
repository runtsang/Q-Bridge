"""AutoencoderGen048: an extended classical autoencoder with sparse and clustering regularization.

This module defines the AutoencoderGen048 class, which extends a standard MLP autoencoder with
additional regularization terms and a simple clustering loss on the latent space. The training
pipeline supports early stopping, learning rate scheduling, and optional KMeans clustering of
the latent codes. The implementation uses PyTorch and can run on CPU or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

@dataclass
class AutoencoderGen048Config:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    l1_weight: float = 0.0
    cluster_weight: float = 0.0
    num_clusters: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 200
    early_stop_patience: int = 10
    device: Optional[torch.device] = None

class AutoencoderGen048(nn.Module):
    """A multilayer perceptron autoencoder with optional sparsity and clustering regularization."""

    def __init__(self, cfg: AutoencoderGen048Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self._build_mlp(cfg.input_dim, cfg.hidden_dims, cfg.latent_dim, cfg.dropout)
        self.decoder = self._build_mlp(cfg.latent_dim, cfg.hidden_dims[::-1], cfg.input_dim, cfg.dropout)
        self.to(self.device)

    def _build_mlp(self, in_dim: int, hidden_dims: Tuple[int,...], out_dim: int, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def _l1_regularizer(self, z: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(z))

    def _cluster_loss(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.cluster_weight == 0.0:
            return torch.tensor(0.0, device=self.device)
        kmeans = KMeans(n_clusters=self.cfg.num_clusters).fit(z.cpu().detach().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, device=self.device)
        labels = torch.tensor(kmeans.labels_, device=self.device)
        loss = torch.mean(torch.norm(z - centers[labels], dim=1))
        return loss

    def train_autoencoder(
        self,
        data: Iterable[float] | torch.Tensor,
        *,
        verbose: bool = True,
    ) -> list[float]:
        """Train the autoencoder on the provided data.

        Parameters
        ----------
        data: Iterable[float] | torch.Tensor
            Training data. Will be converted to a 2‑D float32 tensor.
        verbose: bool
            If True, prints epoch statistics.

        Returns
        -------
        history: list[float]
            Reconstruction loss history.
        """
        x = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        best_loss = float("inf")
        patience = 0

        for epoch in range(self.cfg.epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self(batch)
                loss = loss_fn(recon, batch)
                if self.cfg.l1_weight > 0.0:
                    loss += self.cfg.l1_weight * self._l1_regularizer(self.encode(batch))
                if self.cfg.cluster_weight > 0.0:
                    loss += self.cfg.cluster_weight * self._cluster_loss(self.encode(batch))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss:.6f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
                if patience >= self.cfg.early_stop_patience:
                    if verbose:
                        print("Early stopping triggered")
                    break
        return history

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation of the input."""
        return self.encode(x.to(self.device))

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent codes."""
        return self.decode(z.to(self.device))

    def visualize_latent(self, x: torch.Tensor) -> None:
        """Plot the latent space using matplotlib (2‑D only)."""
        import matplotlib.pyplot as plt
        z = self.encode_latent(x).cpu().detach().numpy()
        if z.shape[1]!= 2:
            raise ValueError("Latent dimension must be 2 for visualization.")
        plt.scatter(z[:, 0], z[:, 1], s=10)
        plt.title("Latent space")
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.show()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, cfg: AutoencoderGen048Config, path: str) -> "AutoencoderGen048":
        model = cls(cfg)
        model.load_state_dict(torch.load(path, map_location=cfg.device))
        return model

__all__ = ["AutoencoderGen048", "AutoencoderGen048Config"]
