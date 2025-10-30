from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import surrogate quantum nets
try:
    from SamplerQNN import SamplerQNN
    from EstimatorQNN import EstimatorQNN
except Exception:
    # Lightweight stubs if modules not present
    class SamplerQNN(nn.Module):
        def __init__(self): super().__init__()
    class EstimatorQNN(nn.Module):
        def __init__(self): super().__init__()

@dataclass
class AutoencoderHybridConfig:
    """Configuration for the hybrid auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_sampler: bool = True   # enable latent sampling regularisation
    use_estimator: bool = True # enable property prediction regularisation

class AutoencoderHybrid(nn.Module):
    """Hybrid classical auto‑encoder with optional quantum surrogate nets."""
    def __init__(self, config: AutoencoderHybridConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(config.dropout)])
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(config.dropout)])
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Auxiliary quantum‑style networks
        self.sampler = SamplerQNN() if config.use_sampler else None
        self.estimator = EstimatorQNN() if config.use_estimator else None

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # ------------------------------------------------------------------
    def _latent_regularisation(self, z: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=z.device)
        if self.sampler is not None:
            probs = self.sampler(z)
            loss += F.kl_div(probs.log(), torch.full_like(probs, 1.0 / probs.shape[-1]),
                             reduction='batchmean')
        if self.estimator is not None:
            preds = self.estimator(z)
            loss += F.mse_loss(preds, torch.zeros_like(preds))
        return loss

    # ------------------------------------------------------------------
    def training_step(self, batch: torch.Tensor, optimizer: torch.optim.Optimizer,
                      loss_fn=nn.MSELoss()) -> torch.Tensor:
        optimizer.zero_grad()
        recon = self(batch)
        rec_loss = loss_fn(recon, batch)
        reg_loss = self._latent_regularisation(self.encode(batch))
        loss = rec_loss + 1e-3 * reg_loss
        loss.backward()
        optimizer.step()
        return loss

def _as_tensor(data: torch.Tensor | list[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(dtype=torch.float32)
    return torch.as_tensor(data, dtype=torch.float32)

def train_autoencoder_hybrid(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> list[float]:
    """Train the hybrid auto‑encoder and return the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = model.training_step(batch, optimizer)
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["AutoencoderHybridConfig", "AutoencoderHybrid", "train_autoencoder_hybrid"]
