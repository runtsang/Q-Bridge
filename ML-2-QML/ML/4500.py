"""Hybrid convolutional‑autoencoder network with a quantum expectation head.

The class `QCNNGen184` combines ideas from the original QCNN, a fully
connected layer (FCL), a lightweight autoencoder, and a quantum hybrid
layer.  All sub‑modules are fully differentiable and can be trained
jointly with standard PyTorch optimisers.

The quantum circuit is injected via the `quantum_circuit` argument,
allowing the user to swap in any `qiskit`‑compatible circuit that
exposes a `run` method returning a single expectation value.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Autoencoder utilities
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class AutoencoderConfig:
    """Configuration for the encoder/decoder MLP."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory that returns a configured autoencoder."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# 2. Fully‑connected layer (FCL) – classical stand‑in
# --------------------------------------------------------------------------- #
class FCL(nn.Module):
    """A simple linear layer that mimics the quantum fully‑connected layer."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """Return a single expectation‑like value."""
        theta = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(theta)).mean(dim=0)


# --------------------------------------------------------------------------- #
# 3. Hybrid quantum interface
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper that forwards activations through a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        # Run the quantum circuit on the CPU for simplicity
        expectation = circuit.run(inputs.tolist())
        result = torch.tensor([expectation], dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        # Finite‑difference gradient estimate
        grads = []
        for val in inputs.detach().numpy():
            right = ctx.circuit.run([val + shift])
            left = ctx.circuit.run([val - shift])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid layer that applies a quantum circuit to the last feature."""
    def __init__(self, circuit, shift: float = 0.0):
        super().__init__()
        self.circuit = circuit
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect a 1‑d tensor of activations
        return HybridFunction.apply(x.squeeze(), self.circuit, self.shift)


# --------------------------------------------------------------------------- #
# 4. Convolutional front‑end (QCNet‑style)
# --------------------------------------------------------------------------- #
class QCNet(nn.Module):
    """Convolutional backbone that mirrors the original QCNN."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        return x


# --------------------------------------------------------------------------- #
# 5. The full hybrid network
# --------------------------------------------------------------------------- #
class QCNNGen184(nn.Module):
    """Hybrid convolution‑autoencoder network with a quantum head.

    Parameters
    ----------
    quantum_circuit : object
        Any object exposing a ``run(list[float]) -> float`` method.
    quantum_shift : float, optional
        Shift used in the finite‑difference gradient estimate.
    """
    def __init__(self, quantum_circuit, quantum_shift: float = 0.0) -> None:
        super().__init__()
        self.backbone = QCNet()
        # The latent dimension is chosen to match the QCNN feature size (4)
        self.autoencoder = Autoencoder(input_dim=4, latent_dim=4, hidden_dims=(8, 4))
        self.fcl = FCL(n_features=1)
        self.hybrid = Hybrid(quantum_circuit, shift=quantum_shift)

        # Final linear layer that maps latent + conv features to a scalar
        self.final_linear = nn.Linear(4 + 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        features = self.backbone(x)
        flat = torch.flatten(features, 1)          # shape (B, 4)
        # Autoencoder regularisation
        latent = self.autoencoder.encode(flat)
        recon = self.autoencoder.decode(latent)
        # Combine original and reconstructed features
        combined = torch.cat([flat, recon], dim=1)  # (B, 8)
        # Linear projection
        logits = self.final_linear(combined).squeeze(-1)
        # Quantum expectation head
        prob = self.hybrid(logits)
        return torch.cat([prob, 1 - prob], dim=-1)


def QCNNGen184_factory(quantum_circuit, quantum_shift: float = 0.0) -> QCNNGen184:
    """Convenience factory that returns a configured QCNNGen184 instance."""
    return QCNNGen184(quantum_circuit, quantum_shift=quantum_shift)


__all__ = [
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "FCL",
    "Hybrid",
    "HybridFunction",
    "QCNet",
    "QCNNGen184",
    "QCNNGen184_factory",
]
