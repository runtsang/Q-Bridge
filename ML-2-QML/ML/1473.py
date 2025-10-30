import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Iterable

from Autoencoder_qml import QuantumDecoder

class Autoencoder__gen030(nn.Module):
    """
    Hybrid classical-quantum autoencoder.

    The encoder is a standard feed‑forward network producing a latent vector.
    The decoder is a quantum circuit implemented in PennyLane that maps the
    latent vector to a probability distribution over the input dimension.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int,...] = (128, 64),
                 dropout: float = 0.1,
                 n_decoder_layers: int = 1,
                 seed: int = 42) -> None:
        super().__init__()

        # Classical encoder
        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum decoder
        self.decoder = QuantumDecoder(num_qubits=input_dim,
                                      latent_dim=latent_dim,
                                      n_layers=n_decoder_layers,
                                      seed=seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)

def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

def train_autoencoder(model: nn.Module,
                      data: torch.Tensor,
                      *,
                      epochs: int = 100,
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      weight_decay: float = 0.0,
                      device: torch.device | None = None) -> list[float]:
    """Train the hybrid autoencoder.

    Uses MSE loss between the quantum decoder output and the input data.
    """
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

__all__ = ["Autoencoder__gen030", "train_autoencoder"]
