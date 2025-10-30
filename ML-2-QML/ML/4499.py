from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, Iterable

# Quantum encoder import
try:
    from.qml_module import HybridAutoencoder as QuantumHybridAutoencoder
except Exception:
    # Provide a placeholder if quantum module is missing
    class QuantumHybridAutoencoder(nn.Module):
        def __init__(self, *_, **__):
            raise ImportError("Quantum encoder module not available.")
        def forward(self, x):
            return x

@dataclass
class HybridAutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    num_qubits: int = 4
    shots: int = 1024

class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder combining a QCNN quantum encoder with a classical decoder."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        # Quantum encoder
        self.quantum_encoder = QuantumHybridAutoencoder(
            latent_dim=config.latent_dim,
            shots=config.shots
        )
        # Classical decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in config.hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs via the quantum encoder."""
        return self.quantum_encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors via the classical decoder."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass."""
        latents = self.encode(inputs)
        return self.decode(latents)

def HybridAutoencoderFactory(
    input_dim: int,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    num_qubits: int = 4,
    shots: int = 1024,
) -> HybridAutoencoder:
    """Convenience factory for the hybrid autoencoder."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_qubits=num_qubits,
        shots=shots,
    )
    return HybridAutoencoder(config)

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Utility to convert data to a float32 tensor."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

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
    """Training loop for the hybrid autoencoder."""
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
    "HybridAutoencoderConfig",
]
