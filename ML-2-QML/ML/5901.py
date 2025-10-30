import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

@dataclass
class AutoencoderConfig:
    """Configuration of the classical encoder‑decoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """A lightweight MLP autoencoder with an optional latent callback."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Build encoder
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Build decoder
        dec_layers = []
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

class Autoencoder__gen303:
    """Hybrid‑style autoencoder that exposes classical training and latent‑space utilities."""

    def __init__(self, config: AutoencoderConfig):
        self.model = AutoencoderNet(config)
        self.config = config

    def train(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ) -> List[float]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.6f}")
        return history

    def evaluate(self, data: torch.Tensor, device: Optional[torch.device] = None) -> float:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            data_t = _as_tensor(data).to(device)
            recon = self.model(data_t)
            loss = nn.MSELoss()(recon, data_t).item()
        return loss

    def plot_latent(self, data: torch.Tensor, n_components: int = 2, cmap: str = "viridis") -> None:
        """Project latent embeddings to 2‑D and scatter‑plot."""
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(_as_tensor(data)).cpu().numpy()
        if n_components == 2:
            pc = PCA(n_components=2).fit_transform(z)
        else:
            pc = z
        plt.figure(figsize=(6, 5))
        plt.scatter(pc[:, 0], pc[:, 1], cmap=cmap, s=20)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Latent space projection")
        plt.tight_layout()
        plt.show()

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = ["Autoencoder__gen303", "AutoencoderConfig", "AutoencoderNet"]
