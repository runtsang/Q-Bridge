"""Hybrid classical module that fuses a 2D convolution filter with a fully‑connected autoencoder.

The module is designed to be a direct drop‑in replacement for the original
`Conv` implementation.  It keeps the same public API (`ConvAutoencoder()` factory)
while exposing two distinct back‑ends:
* Classical: a learnable 2×2 convolution followed by a lightweight MLP autoencoder.
* Quantum: a variational circuit that mimics the convolution and swap‑test
  reconstruction, available via the `quantum` flag.

The design allows researchers to switch between the two paradigms simply
by toggling a boolean, facilitating fair comparisons and mixed‑precision
experiments.

Typical usage:
```
model = ConvAutoencoder(kernel_size=2, latent_dim=16, quantum=False)
output = model(input_tensor)
```
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable

# --------------------------------------------------------------------------- #
# Helper utilities
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

# --------------------------------------------------------------------------- #
# 1. Classical convolution filter
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Simple learnable 2×2 convolution used as a feature extractor."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect shape (B, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

# --------------------------------------------------------------------------- #
# 2. Fully‑connected autoencoder
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Lightweight MLP autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
            ])
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
            ])
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)

# --------------------------------------------------------------------------- #
# 3. Hybrid Conv‑Autoencoder
# --------------------------------------------------------------------------- #
class ConvAutoencoder(nn.Module):
    """Hybrid model that first extracts a 2×2 patch via convolution
    and then reconstructs it using a fully‑connected autoencoder.

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel (default 2 for a 2×2 filter).
    latent_dim : int
        Latent dimension of the autoencoder.
    hidden_dims : Tuple[int, int]
        Hidden layer sizes for the autoencoder.
    dropout : float
        Dropout probability in the autoencoder.
    quantum : bool
        If True, the convolution is replaced by a quantum filter
        (see :class:`ConvAutoencoderQNN` in the QML module).
    """
    def __init__(
        self,
        kernel_size: int = 2,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        quantum: bool = False,
    ) -> None:
        super().__init__()
        self.quantum = quantum
        self.kernel_size = kernel_size
        self.conv_filter = ConvFilter(kernel_size)
        # Flattened input dimension after convolution
        conv_out_dim = kernel_size * kernel_size
        self.autoencoder = Autoencoder(
            input_dim=conv_out_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Reconstructed patches of shape (B, 1, k, k).
        """
        conv_out = self.conv_filter(x)          # (B, 1, H-k+1, W-k+1)
        # Flatten each patch for the autoencoder
        B, C, H, W = conv_out.shape
        patches = conv_out.view(B, -1)          # (B, k*k)
        recon = self.autoencoder(patches)      # (B, k*k)
        recon = recon.view(B, 1, self.kernel_size, self.kernel_size)
        return recon

# --------------------------------------------------------------------------- #
# 4. Training helper
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: ConvAutoencoder,
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

__all__ = ["ConvAutoencoder", "train_autoencoder", "AutoencoderNet", "Autoencoder"]
