"""Hybrid estimator combining classical neural networks, autoencoders, and fully connected layers.

The estimator supports deterministic evaluation of a PyTorch model and optional Gaussian shot noise.
It can optionally prepend a classical autoencoder or fully connected layer to the input pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class AutoencoderConfig:
    """Configuration for a simple fully‑connected autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """Lightweight multilayer perceptron autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
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


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_ensure_batch(data))
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


class FastBaseEstimator:
    """Hybrid estimator for classical neural networks.

    Parameters
    ----------
    model
        The primary PyTorch model.  If ``None`` a dummy identity model is used.
    autoencoder
        Optional autoencoder applied to the input parameters before the model.
    fcl
        Optional fully‑connected layer applied to the raw parameters.
    """
    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        autoencoder: nn.Module | None = None,
        fcl: nn.Module | None = None,
    ) -> None:
        self.model = model or nn.Identity()
        self.autoencoder = autoencoder
        self.fcl = fcl

    def _preprocess(self, params: torch.Tensor) -> torch.Tensor:
        """Apply optional FCL and autoencoder to the input parameters."""
        if self.fcl is not None:
            params = self.fcl(params)
        if self.autoencoder is not None:
            params = self.autoencoder.encode(params)
        return params

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Deterministic evaluation of the wrapped model."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                pre = self._preprocess(inputs)
                outputs = self.model(pre)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().cpu()
                    row.append(float(val))
                results.append(row)
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate with optional Gaussian shot noise."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @staticmethod
    def Autoencoder(
        input_dim: int,
        *,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> AutoencoderNet:
        """Return a configured autoencoder."""
        cfg = AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        return AutoencoderNet(cfg)

    @staticmethod
    def FCL(n_features: int = 1) -> nn.Module:
        """Return a classical fully‑connected layer mimicking the quantum example."""
        class FullyConnectedLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(n_features, 1)

            def forward(self, thetas: torch.Tensor) -> torch.Tensor:
                return torch.tanh(self.linear(thetas)).mean(dim=0)

        return FullyConnectedLayer()


__all__ = ["FastBaseEstimator", "AutoencoderNet", "AutoencoderConfig", "train_autoencoder"]
