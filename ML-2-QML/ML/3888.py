from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple

def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    """Convert inputs to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class HybridQCNN(nn.Module):
    """
    Hybrid QCNN‑autoencoder that fuses a QCNN‑style encoder with a classic
    autoencoder decoder.  The encoder reduces the input to a latent vector
    of size ``latent_dim``; the decoder reconstructs the input.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int,...] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Encoder – QCNN‑style linear stack
        encoder_layers = [
            nn.Linear(input_dim, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8),  nn.Tanh(),
            nn.Linear(8, 4),   nn.Tanh(),
            nn.Linear(4, 4),   nn.Tanh(),
            nn.Linear(4, latent_dim),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder – standard autoencoder MLP
        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from the latent vector."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode → decode."""
        return self.decode(self.encode(x))

def HybridQCNNModel(
    input_dim: int,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
) -> HybridQCNN:
    """Return a ready‑to‑train HybridQCNN instance."""
    return HybridQCNN(input_dim, latent_dim, hidden_dims, dropout)

def train_hybrid_autoencoder(
    model: HybridQCNN,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train the hybrid autoencoder and return the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = []

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

__all__ = ["HybridQCNN", "HybridQCNNModel", "train_hybrid_autoencoder"]
