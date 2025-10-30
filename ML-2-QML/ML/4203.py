"""Hybrid classical autoencoder with quantum kernel regularization and convolutional feature extractor.

This module defines a shared class `AutoencoderHybridNet` that can be used as a drop‑in
replacement for the original `Autoencoder` while adding two quantum‑aware extensions:

1. A lightweight 2×2 convolutional layer (adapted from the `ConvFilter` in *Conv.py*)
   that extracts local patterns before feeding the flattened vector into the
   fully‑connected encoder.
2. A radial‑basis‑function quantum kernel (adapted from the TorchQuantum `Kernel` in
   *QuantumKernelMethod.py*) that is evaluated on the latent representation.  The
   kernel value is used as an additional regulariser in `train_autoencoder_hybrid`.

The implementation stays fully classical – all operations are performed with
PyTorch tensors and CPU/GPU back‑ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# 1. Convolutional feature extractor (quantum inspired)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """
    Lightweight 2×2 convolutional filter with a learnable bias.
    The filter is applied to the input before flattening.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., H, W).  The tensor is reshaped to
            (batch, 1, H, W) before the convolution.

        Returns
        -------
        torch.Tensor
            Output after sigmoid activation and mean pooling.
        """
        tensor = x.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])

# --------------------------------------------------------------------------- #
# 2. Quantum kernel (RBF) implemented with TorchQuantum
# --------------------------------------------------------------------------- #
class QuantumRBFKernel(nn.Module):
    """
    Radial‑basis‑function kernel evaluated on a fixed quantum ansatz.
    The kernel is implemented using TorchQuantum to keep the interface
    compatible with the classical version.
    """
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        import torchquantum as tq
        from torchquantum.functional import func_name_dict
        self.tq = tq
        self.func_name_dict = func_name_dict
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        # Simple Ry rotation ansatz
        self.ansatz = [
            {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)
        ]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute |⟨ψ(x)|ψ(y)⟩|² for two input vectors.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.q_device.reset_states(x.shape[0])
        # Encode x
        for info in self.ansatz:
            params = x[:, info["input_idx"]]
            self.func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        # Encode y with negative parameters (reverse)
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]]
            self.func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        # Return absolute overlap
        return torch.abs(self.q_device.states.view(-1)[0])

# --------------------------------------------------------------------------- #
# 3. Hybrid autoencoder
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderHybridConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    conv_kernel: int = 2

class AutoencoderHybridNet(nn.Module):
    """
    Hybrid autoencoder that combines a convolutional feature extractor,
    a fully‑connected encoder/decoder, and a quantum kernel regulariser.
    """
    def __init__(self, cfg: AutoencoderHybridConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 1. Convolutional layer
        self.conv = ConvFilter(kernel_size=cfg.conv_kernel)

        # 2. Encoder
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # 3. Decoder (mirrored)
        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # 4. Quantum kernel for regularisation
        self.kernel = QuantumRBFKernel(n_wires=cfg.latent_dim)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply convolution + encoder."""
        conv_out = self.conv(inputs)
        return self.encoder(conv_out)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def AutoencoderHybrid(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    conv_kernel: int = 2,
) -> AutoencoderHybridNet:
    """Factory that returns a configured hybrid autoencoder."""
    cfg = AutoencoderHybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        conv_kernel=conv_kernel,
    )
    return AutoencoderHybridNet(cfg)

# --------------------------------------------------------------------------- #
# 4. Training helper with optional quantum kernel regularisation
# --------------------------------------------------------------------------- #
def train_autoencoder_hybrid(
    model: AutoencoderHybridNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    kernel_weight: float = 0.0,
) -> list[float]:
    """
    Train the hybrid autoencoder.  If ``kernel_weight`` > 0, the loss is
    augmented by a quantum‑kernel based similarity term that encourages
    latent vectors of similar inputs to be close in the quantum feature space.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    recon_loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            latent = model.encode(batch)
            recon = model.decode(latent)
            loss = recon_loss_fn(recon, batch)

            if kernel_weight > 0.0:
                # pairwise kernel between latent vectors
                k = model.kernel(latent, latent)
                # encourage values close to 1 (high similarity)
                loss += kernel_weight * (1.0 - k.mean())

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "AutoencoderHybrid",
    "AutoencoderHybridNet",
    "AutoencoderHybridConfig",
    "train_autoencoder_hybrid",
]
