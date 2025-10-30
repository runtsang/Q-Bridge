"""Hybrid classical auto‑encoder that mimics quantum layers with classical surrogates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: torch.Tensor | list[float]) -> torch.Tensor:
    """Ensure input is a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


class ConvFilter(nn.Module):
    """Simple 2‑D convolutional filter used as a drop‑in replacement for a quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2, 3))  # collapse spatial dims


class QuantumAutoEncoderSurrogate(nn.Module):
    """Classical surrogate that approximates a parameterised quantum auto‑encoder."""
    def __init__(self, latent_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class ClassicalQLSTM(nn.Module):
    """Drop‑in replacement for a quantum LSTM using classical linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = inputs.size(1)
        hx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)


@dataclass
class HybridAutoencoderConfig:
    """Configuration for the hybrid auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    conv_kernel: int = 2
    conv_threshold: float = 0.0
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_quantum_ae: bool = False
    use_quantum_lstm: bool = False


class HybridAutoencoder(nn.Module):
    """Classical hybrid auto‑encoder that optionally replaces the encoder/decoder
    and the LSTM with quantum surrogates."""
    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = ConvFilter(kernel_size=cfg.conv_kernel, threshold=cfg.conv_threshold)

        # Encoder
        enc_layers = []
        in_dim = cfg.input_dim * cfg.conv_kernel * cfg.conv_kernel
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Optional quantum auto‑encoder surrogate
        self.quantum_ae = QuantumAutoEncoderSurrogate(cfg.latent_dim) if cfg.use_quantum_ae else None

        # LSTM layer
        if cfg.use_quantum_lstm:
            self.lstm = ClassicalQLSTM(cfg.latent_dim, cfg.latent_dim)
        else:
            self.lstm = nn.LSTM(cfg.latent_dim, cfg.latent_dim, batch_first=True)

        # Decoder
        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim * cfg.conv_kernel * cfg.conv_kernel))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W)
        conv_out = self.conv(x).view(x.size(0), -1)
        latent = self.encoder(conv_out)

        if self.quantum_ae is not None:
            latent = self.quantum_ae(latent)

        # LSTM expects (seq_len, batch, input_size) if batch_first=False
        latent_seq = latent.unsqueeze(0)  # seq_len = 1
        lstm_out, _ = self.lstm(latent_seq)
        lstm_out = lstm_out.squeeze(0)

        recon = self.decoder(lstm_out)
        recon = recon.view(x.size())
        return recon


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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["HybridAutoencoder", "HybridAutoencoderConfig", "train_hybrid_autoencoder"]
