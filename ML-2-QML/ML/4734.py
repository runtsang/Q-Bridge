"""Hybrid classical-quantum autoencoder combining a fully-connected backbone and a quantum kernel regularizer."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Optional, Sequence

# Optional quantum kernel imports
try:
    import torchquantum as tq
except Exception:
    tq = None

def _as_tensor(data: Sequence[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_quantum_kernel: bool = False
    gamma: float = 1.0

class QuantumRBFKernel(nn.Module):
    """Quantum RBF kernel using TorchQuantum or classical fallback."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        if tq is None:
            self.use_classical = True
        else:
            self.use_classical = False
            self.n_wires = 4
            self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
            # Build a simple ansatz that encodes each feature into a ry gate
            def forward(q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
                q_device.reset_states(x.shape[0])
                # encode x
                for i in range(self.n_wires):
                    q_device.ry(x[:, i], wires=[i])
                # encode -y
                for i in range(self.n_wires):
                    q_device.ry(-y[:, i], wires=[i])
            self.ansatz = tq.QuantumModule()
            self.ansatz.forward = forward

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_classical:
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
        # quantum evaluation
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(0)

class HybridAutoencoderNet(nn.Module):
    """Classical encoder-decoder with optional quantum kernel regularizer and a small refinement network."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
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

        # Small refinement network inspired by EstimatorQNN
        self.refine = nn.Sequential(
            nn.Linear(config.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, config.input_dim),
        )

        self.use_quantum_kernel = config.use_quantum_kernel
        if self.use_quantum_kernel:
            self.kernel = QuantumRBFKernel(config.gamma)
        else:
            self.kernel = None

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.decode(self.encode(inputs))
        return self.refine(x)

def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_quantum_kernel: bool = False,
    gamma: float = 1.0,
) -> HybridAutoencoderNet:
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_quantum_kernel=use_quantum_kernel,
        gamma=gamma,
    )
    return HybridAutoencoderNet(config)

def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    lambda_kernel: float = 0.1,
) -> list[float]:
    """Training loop that optionally adds a quantum kernel regularizer."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    recon_loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            recon_loss = recon_loss_fn(recon, batch)

            if model.use_quantum_kernel and model.kernel is not None:
                latents = model.encode(batch)
                kernel_vals = model.kernel(latents, latents)
                kernel_loss = -kernel_vals.mean()
                loss = recon_loss + lambda_kernel * kernel_loss
            else:
                loss = recon_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridAutoencoder", "HybridAutoencoderConfig", "train_hybrid_autoencoder"]
