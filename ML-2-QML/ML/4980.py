"""Hybrid classical autoencoder combining MLP, convolutional filter, and sampler network.

The module implements a lightweight neural network that can be trained on
classical data.  It reuses the design patterns from the reference seeds:
* fully‑connected encoder/decoder (AutoencoderNet)
* 2‑D convolutional feature extractor (ConvFilter)
* simple sampler‑based classifier (SamplerModule)
* estimator utilities (FastEstimator) for shot‑noise simulation.

The class `AutoencoderGen227` can be instantiated with a configuration
object and used as a drop‑in replacement for the original `Autoencoder`
in downstream pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    conv_kernel: int = 2
    conv_threshold: float = 0.0

# --------------------------------------------------------------------------- #
# Classic building blocks
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """2‑D convolutional filter emulating a quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray | torch.Tensor) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class SamplerModule(nn.Module):
    """A lightweight sampler‑based network for classification or loss."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

# --------------------------------------------------------------------------- #
# Autoencoder core
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """Standard fully‑connected autoencoder."""
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

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

# --------------------------------------------------------------------------- #
# Hybrid wrapper
# --------------------------------------------------------------------------- #
class AutoencoderGen227(nn.Module):
    """
    Hybrid autoencoder that combines a convolutional feature extractor,
    the core MLP encoder/decoder, and a sampler network.
    """
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.conv = ConvFilter(kernel_size=config.conv_kernel,
                               threshold=config.conv_threshold)
        self.autoencoder = AutoencoderNet(config)
        self.sampler = SamplerModule()

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the hybrid network.

        Returns:
            reconstruction: decoded output
            sampler_output: probability distribution from sampler
        """
        # Pass through conv filter (treated as a non‑trainable feature extractor)
        conv_feat = torch.tensor(self.conv.run(inputs.numpy()),
                                 dtype=torch.float32,
                                 device=inputs.device).unsqueeze(0)
        # Encode‑decode
        latent = self.autoencoder.encode(inputs)
        reconstruction = self.autoencoder.decode(latent)
        # Sampler network
        sampler_out = self.sampler(conv_feat)
        return reconstruction, sampler_out

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(latents)

# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def train_autoencoder_gen227(
    model: AutoencoderGen227,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """
    Train the hybrid autoencoder on the supplied data.

    The loss is a weighted sum of reconstruction error and sampler entropy
    to encourage meaningful latent representations.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_ensure_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, sampler_out = model(batch)
            recon_loss = loss_fn(recon, batch)
            # Encourage sampler diversity
            entropy = -(sampler_out * sampler_out.log()).sum(dim=-1).mean()
            loss = recon_loss + 0.01 * entropy
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
# Estimator utilities
# --------------------------------------------------------------------------- #
def _ensure_tensor(values: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Ensure values are a float32 tensor on the current device."""
    if isinstance(values, torch.Tensor):
        tensor = values
    else:
        tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class FastEstimator:
    """
    Estimator that adds Gaussian shot‑noise to a deterministic model
    evaluation, mimicking quantum measurement statistics.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_tensor(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["AutoencoderGen227", "AutoencoderConfig", "train_autoencoder_gen227", "FastEstimator"]
