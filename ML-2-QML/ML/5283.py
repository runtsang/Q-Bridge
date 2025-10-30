"""Hybrid kernel, autoencoder, and LSTM model with optional quantum backends.

The module defines a single :class:`HybridKernelModel` class that can operate entirely
classically (using PyTorch) or switch to quantum components via flags.  The
class exposes a RBF kernel, a fully‑connected autoencoder, a feed‑forward
classifier, and a quantum‑or‑classical LSTM for sequence tagging.  All
sub‑components are compatible with the original seed API, and the public
interface matches the quantum variant for downstream tooling.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Classical RBF kernel
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial‑basis function kernel implemented with PyTorch."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of tensors."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 2. Auto‑encoder
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for the fully‑connected auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Multi‑layer perceptron auto‑encoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
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
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)

# --------------------------------------------------------------------------- #
# 3. Classical classifier factory
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a feed‑forward classifier and metadata similar to the quantum variant."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding: List[int] = list(range(num_features))
    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables: List[int] = list(range(2))
    return network, encoding, weight_sizes, observables

# --------------------------------------------------------------------------- #
# 4. Classical / Quantum LSTM
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Drop‑in replacement using classical linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# --------------------------------------------------------------------------- #
# 5. Hybrid model
# --------------------------------------------------------------------------- #
class HybridKernelModel(nn.Module):
    """Unified model that supports classical or quantum back‑ends.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vectors.
    latent_dim : int, default 32
        Size of the latent space in the auto‑encoder.
    hidden_dims : Tuple[int, int], default (128, 64)
        Hidden layer sizes for the auto‑encoder.
    dropout : float, default 0.1
        Drop‑out probability in the auto‑encoder.
    kernel_gamma : float, default 1.0
        RBF kernel bandwidth.
    depth : int, default 2
        Depth of the feed‑forward classifier.
    n_qubits : int, default 4
        Number of qubits used if a quantum LSTM is requested.
    use_quantum_lstm : bool, default False
        Whether to replace the classical LSTM with a quantum variant.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        kernel_gamma: float = 1.0,
        depth: int = 2,
        n_qubits: int = 4,
        use_quantum_lstm: bool = False,
    ) -> None:
        super().__init__()
        self.kernel = RBFKernel(kernel_gamma)
        self.autoencoder = Autoencoder(
            input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            input_dim, depth
        )
        if use_quantum_lstm:
            self.lstm = QLSTM(input_dim, hidden_dim=latent_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim=latent_dim)

    def kernel_matrix(self, X: Iterable[torch.Tensor], Y: Iterable[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix between two collections of tensors."""
        return np.array([[self.kernel(x, y).item() for y in Y] for x in X])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, classify and optionally run through an LSTM."""
        z = self.autoencoder.encode(x)
        logits = self.classifier(z)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(z.unsqueeze(0))
            logits = self.classifier(lstm_out.squeeze(0))
        else:
            lstm_out, _ = self.lstm(z.unsqueeze(0))
            logits = self.classifier(lstm_out.squeeze(0))
        return logits

__all__ = [
    "HybridKernelModel",
    "RBFKernel",
    "kernel_matrix",
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "build_classifier_circuit",
    "QLSTM",
]
