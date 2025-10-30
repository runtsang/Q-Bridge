"""Hybrid classical autoencoder with convolutional encoder and optional quantum latent embedding.

The module defines:
- ConvFilter: lightweight conv encoder used to extract image patches.
- QuanvolutionFilter: classical analogue of a quantum convolution filter.
- AutoencoderGen010: a PyTorch nn.Module that chains ConvFilter, a linear encoder,
  optional quantum encoder, and a linear decoder.
- train_autoencoder: generic training loop that supports a quantum encoder callback.

The design follows the original Autoencoder.py but extends it with a first
convolutional layer and a hook for quantum latent manipulation, enabling
co‑training with the quantum implementation in the QML module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderGen010`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    conv_kernel_size: int = 2
    conv_threshold: float = 0.0
    use_quantum: bool = False


class ConvFilter(nn.Module):
    """A lightweight convolutional feature extractor.

    Mirrors the behaviour of the classical Conv implementation from
    ``Conv.py`` but exposes a :meth:`forward` interface compatible with
    :class:`torch.nn.Module`.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # Reduce spatial dims to a single scalar per sample
        return activations.mean(dim=[2, 3], keepdim=True)


class QuanvolutionFilter(nn.Module):
    """Classical analogue of a quantum convolution filter.

    Inspired by ``Quanvolution.py`` – the filter produces a flattened
    feature vector from a batch of grayscale images.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class AutoencoderGen010(nn.Module):
    """Hybrid autoencoder combining a ConvFilter, linear encoder/decoder
    and an optional quantum latent encoder.
    """
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        # Convolutional encoder
        self.conv = ConvFilter(
            kernel_size=config.conv_kernel_size,
            threshold=config.conv_threshold,
        )
        # Flatten after conv
        conv_out_dim = 1  # because ConvFilter reduces to scalar per sample
        # Linear encoder
        encoder_layers = []
        in_dim = conv_out_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Linear decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, conv_out_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Optional quantum encoder placeholder
        self.quantum_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def set_quantum_encoder(
        self,
        quantum_encoder: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Attach a quantum encoder that transforms the latent vector
        before decoding.
        """
        self.quantum_encoder = quantum_encoder

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply convolution and linear encoder."""
        x = inputs.view(-1, 1, *self._input_shape(inputs))
        conv_out = self.conv(x)
        flat = conv_out.view(conv_out.size(0), -1)
        return self.encoder(flat)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to original shape."""
        if self.quantum_encoder is not None:
            latents = self.quantum_encoder(latents)
        decoded = self.decoder(latents)
        # Reshape to original input shape
        batch_size = decoded.size(0)
        return decoded.view(batch_size, *self._input_shape(decoded))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

    def _input_shape(self, tensor: torch.Tensor) -> Tuple[int, int]:
        """Infer spatial dimensions from input tensor."""
        # Assume input is (batch, channels, height, width)
        if tensor.dim() == 4:
            return tensor.shape[2], tensor.shape[3]
        raise ValueError("Input tensor must be 4‑D (B,C,H,W)")

    def _reconstruct(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs)


def train_autoencoder(
    model: AutoencoderGen010,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train the hybrid autoencoder.

    Parameters
    ----------
    model : AutoencoderGen010
        The model to train.
    data : torch.Tensor
        Input data of shape (N, C, H, W).
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Batch size.
    lr : float, optional
        Learning rate.
    weight_decay : float, optional
        Weight decay.
    device : torch.device | None, optional
        Device to run training on; defaults to CUDA if available.

    Returns
    -------
    history : list[float]
        Training loss per epoch.
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
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "AutoencoderConfig",
    "ConvFilter",
    "QuanvolutionFilter",
    "AutoencoderGen010",
    "train_autoencoder",
]
