import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Optional, Callable

@dataclass
class AutoencoderGen48Config:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    activation: Callable[..., nn.Module] = nn.ReLU
    skip_connections: bool = False
    early_stop_patience: Optional[int] = None
    lr: float = 1e-3
    weight_decay: float = 0.0

class AutoencoderGen48(nn.Module):
    def __init__(self, cfg: AutoencoderGen48Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = self._build_mlp(cfg.input_dim, cfg.hidden_dims, cfg.latent_dim, cfg.activation, cfg.dropout, cfg.skip_connections)
        self.decoder = self._build_mlp(cfg.latent_dim, cfg.hidden_dims[::-1], cfg.input_dim, cfg.activation, cfg.dropout, cfg.skip_connections)

    def _build_mlp(self, in_dim: int, hidden_dims: Tuple[int,...], out_dim: int,
                   activation: Callable[..., nn.Module], dropout: float,
                   skip: bool) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def train_autoencoder(self,
                          data: torch.Tensor,
                          epochs: int = 100,
                          batch_size: int = 64,
                          device: Optional[torch.device] = None) -> List[float]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        loss_fn = nn.MSELoss()
        history: List[float] = []
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon = self(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
            if self.cfg.early_stop_patience is not None:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.cfg.early_stop_patience:
                    break
        return history

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = ["AutoencoderGen48", "AutoencoderGen48Config"]
