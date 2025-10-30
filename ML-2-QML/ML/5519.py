from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, List, Optional, Iterable

# Import the quantum refinement class
try:
    from UnifiedAutoencoder_Q import UnifiedAutoencoder as QuantumAutoencoder
except Exception:  # pragma: no cover
    class QuantumAutoencoder(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError("QuantumAutoencoder requires the quantum module.")

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
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
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Standard fully‑connected autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

class UnifiedAutoencoder(nn.Module):
    """Hybrid autoencoder that optionally refines latent vectors with a quantum circuit."""
    def __init__(
        self,
        input_dim: int,
        *,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        use_quantum: bool = False,
        quantum_latent_dim: Optional[int] = None,
        quantum_reps: int = 3,
        quantum_shots: int = 1024,
    ) -> None:
        super().__init__()
        self.config = AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.encoder = AutoencoderNet(self.config)
        self.decoder = nn.Sequential(*list(self.encoder.decoder.children()))
        self.use_quantum = use_quantum
        if use_quantum:
            if quantum_latent_dim is None:
                raise ValueError("quantum_latent_dim must be specified when use_quantum=True")
            self.quantum_refine = QuantumAutoencoder(
                latent_dim=quantum_latent_dim,
                reps=quantum_reps,
                shots=quantum_shots,
            )
            if quantum_latent_dim!= latent_dim:
                self.latent_mapper = nn.Linear(latent_dim, quantum_latent_dim)
            else:
                self.latent_mapper = None
        else:
            self.quantum_refine = None
            self.latent_mapper = None

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder.encode(inputs)
        if self.use_quantum:
            if self.latent_mapper is not None:
                latent = self.latent_mapper(latent)
            refined = self.quantum_refine(latent.detach().cpu().numpy())
            refined = torch.as_tensor(refined, dtype=latent.dtype, device=latent.device)
            return refined
        return latent

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def train_autoencoder(
    model: UnifiedAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    use_quantum_loss: bool = False,
    quantum_loss_weight: float = 0.1,
) -> List[float]:
    """Train the (hybrid) autoencoder, optionally adding a quantum fidelity loss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            if use_quantum_loss and model.use_quantum:
                latent = model.encoder.encode(batch)
                if model.latent_mapper is not None:
                    latent = model.latent_mapper(latent)
                refined = model.quantum_refine(latent.detach().cpu().numpy())
                refined = torch.as_tensor(refined, dtype=latent.dtype, device=latent.device)
                q_loss = loss_fn(refined, latent)
                loss += quantum_loss_weight * q_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "UnifiedAutoencoder",
    "train_autoencoder",
]
