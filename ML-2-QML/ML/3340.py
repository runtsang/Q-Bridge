"""Hybrid estimator combining PyTorch auto‑encoder evaluation and training with optional shot noise.

The implementation mirrors the two source seeds:
*  FastBaseEstimator – efficient batch evaluation and Gaussian noise simulation.
*  Autoencoder – fully‑connected auto‑encoder definition and training loop.

Both are wrapped into a single public class :class:`HybridAutoEncoderEstimator` that can be instantiated with an existing ``nn.Module`` or with an ``AutoencoderConfig`` to build a new auto‑encoder on‑the‑fly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Auto‑encoder definition – adapted from the seed
# --------------------------------------------------------------------------- #

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
    """Configuration for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Lightweight multilayer perceptron auto‑encoder."""
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


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Convenience factory mirroring the original seed."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


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
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
# Hybrid estimator – combines FastBaseEstimator and Autoencoder functionality
# --------------------------------------------------------------------------- #

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class HybridAutoEncoderEstimator:
    """
    Public API that unifies the classical FastBaseEstimator and Autoencoder.
    Parameters
    ----------
    model : nn.Module, optional
        Pre‑trained or empty model. If ``None`` and ``config`` is supplied,
        a new :class:`AutoencoderNet` will be created.
    config : AutoencoderConfig, optional
        Configuration for a new auto‑encoder. Ignored if ``model`` is provided.
    """
    def __init__(self, model: nn.Module | None = None, *, config: AutoencoderConfig | None = None) -> None:
        if model is None:
            if config is None:
                raise ValueError("Either a model or a config must be supplied.")
            self.model = Autoencoder(config.input_dim, latent_dim=config.latent_dim,
                                     hidden_dims=config.hidden_dims, dropout=config.dropout)
        else:
            self.model = model
        self.model.eval()

    # ----------------------------------------------------------- #
    # Evaluation utilities – copied and extended from FastBaseEstimator
    # ----------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate a list of observables on a batch of inputs.
        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns either a tensor or a scalar.
        parameter_sets : sequence of parameter sequences
            Each inner sequence represents a single input to the model.
        shots : int, optional
            If supplied, Gaussian noise with variance 1/shots is added to each result.
        seed : int, optional
            Random seed for the noise generator.
        Returns
        -------
        results : list[list[float]]
            Outer list aligns with ``parameter_sets``, inner list with ``observables``.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        raw_results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                raw_results.append(row)

        if shots is None:
            return raw_results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw_results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    @staticmethod
    def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    # ----------------------------------------------------------- #
    # Training utilities – adapted from Autoencoder.train_autoencoder
    # ----------------------------------------------------------- #
    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
    ) -> list[float]:
        """
        Train the underlying auto‑encoder.  This method is only meaningful if the model
        is an instance of :class:`AutoencoderNet`; otherwise it raises an error.
        """
        if not isinstance(self.model, AutoencoderNet):
            raise TypeError("train_autoencoder can only be called on an AutoencoderNet instance.")
        return train_autoencoder(
            self.model,
            data,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )

    # ----------------------------------------------------------- #
    # Convenience factory – mirrors the original Autoencoder helper
    # ----------------------------------------------------------- #
    @staticmethod
    def build_autoencoder(
        input_dim: int,
        *,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> "HybridAutoEncoderEstimator":
        return HybridAutoEncoderEstimator(
            config=AutoencoderConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )


__all__ = ["HybridAutoEncoderEstimator"]
