import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

__all__ = ["UnifiedAutoencoder", "UnifiedAutoencoderConfig", "train_unified_autoencoder"]

class UnifiedAutoencoderConfig:
    """Configuration for the UnifiedAutoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1, weight_decay: float = 0.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.weight_decay = weight_decay

def _as_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

class ClassicalEncoder(nn.Module):
    def __init__(self, config: UnifiedAutoencoderConfig):
        super().__init__()
        layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.latent_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ClassicalDecoder(nn.Module):
    def __init__(self, config: UnifiedAutoencoderConfig):
        super().__init__()
        layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.input_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

class UnifiedAutoencoder(nn.Module):
    """Classic fully‑connected autoencoder with optional quantum latent path."""
    def __init__(self, config: UnifiedAutoencoderConfig, use_quantum: bool = False):
        super().__init__()
        self.encoder = ClassicalEncoder(config)
        self.decoder = ClassicalDecoder(config)
        self.use_quantum = use_quantum

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def train_unified_autoencoder(
    model: UnifiedAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history
