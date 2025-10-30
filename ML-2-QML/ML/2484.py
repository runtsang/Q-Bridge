import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Import the quantum module (assumed to be named quantum_autoencoder.py)
import quantum_autoencoder

def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
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
    """Configuration values for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class QuantumLayer(nn.Module):
    """Wraps a Qiskit SamplerQNN to act as a differentiable quantum layer."""
    def __init__(self, qnn: quantum_autoencoder.HybridAutoencoder) -> None:
        super().__init__()
        self.qnn = qnn
        self.input_params = list(qnn.qnn.input_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, latent_dim)
        param_dict = {p: x[:, idx].tolist() for idx, p in enumerate(self.input_params)}
        out = self.qnn.qnn.run(param_dict)  # list of numpy arrays
        out_arr = np.stack(out)
        return torch.tensor(out_arr, dtype=torch.float32, device=x.device)

class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder that combines classical dense layers with a variational quantum layer."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.input_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.latent_dim),
        )
        # quantum layer
        qnn = quantum_autoencoder.HybridAutoencoder(config.latent_dim)
        self.quantum_layer = QuantumLayer(qnn)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantum_encode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.quantum_layer(latent)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        q_latent = self.quantum_encode(latent)
        return self.decode(q_latent)

def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoder:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoder(config)

def train_hybrid_autoencoder(
    model: HybridAutoencoder,
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
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
    "train_hybrid_autoencoder",
    "AutoencoderConfig",
]
