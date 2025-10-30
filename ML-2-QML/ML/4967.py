import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from dataclasses import dataclass
from typing import Tuple

def _as_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

class SamplerQNN(nn.Module):
    def __init__(self, input_dim: int = 32, output_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class HybridAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
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

        self.sampler = SamplerQNN(input_dim=config.latent_dim, output_dim=config.latent_dim)

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

        self.regressor = nn.Linear(config.latent_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z = self.sampler(z)
        return self.decode(z)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z = self.sampler(z)
        return self.regressor(z)

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoder:
    cfg = AutoencoderConfig(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims, dropout=dropout)
    return HybridAutoencoder(cfg)

def train_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data), _as_tensor(targets) if targets is not None else torch.zeros_like(_as_tensor(data)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch_x)
            loss = loss_fn(recon, batch_x)
            if targets is not None:
                pred = model.predict(batch_x)
                loss += loss_fn(pred.squeeze(-1), batch_y.squeeze(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# regression dataset utilities
class RegressionDataset(Dataset):
    def __init__(self, samples: int = 1000, num_features: int = 20):
        self.x, self.y = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"states": torch.tensor(self.x[idx], dtype=torch.float32),
                "target": torch.tensor(self.y[idx], dtype=torch.float32)}

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

__all__ = ["Autoencoder", "HybridAutoencoder", "train_autoencoder", "RegressionDataset", "generate_superposition_data", "SamplerQNN"]
