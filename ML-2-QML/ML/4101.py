"""Hybrid classical model combining kernel, LSTM, and autoencoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

# --------------------------------------------------------------------------- #
# Classical kernel components
# --------------------------------------------------------------------------- #
class ClassicalKernelAnsatz(nn.Module):
    """RBF kernel ansatz that accepts a pair of feature vectors."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class ClassicalKernel(nn.Module):
    """Wraps :class:`ClassicalKernelAnsatz` to match the original API."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = ClassicalKernelAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def classical_kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute the Gram matrix between two sets of feature vectors."""
    kernel = ClassicalKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Autoencoder components
# --------------------------------------------------------------------------- #
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with symmetric encoder/decoder."""
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: tuple[int, int] = (128, 64),
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
# Classical LSTM component
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    """A lightweight wrapper around :class:`torch.nn.LSTM`."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.lstm(inputs, states)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return torch.log_softmax(tag_logits, dim=-1)

# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #
class HybridKernelAutoLSTM(nn.Module):
    """Hybrid pipeline that encodes data with an autoencoder, compares them
    with a kernel, and tags sequences with an LSTM."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int = 32,
        gamma: float = 1.0,
        vocab_size: int = 1000,
        tagset_size: int = 10,
    ) -> None:
        super().__init__()
        self.kernel = ClassicalKernel(gamma)
        self.autoencoder = Autoencoder(input_dim, latent_dim=latent_dim)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of feature vectors."""
        return self.autoencoder.encode(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latent vectors."""
        return self.autoencoder.decode(latents)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Compute the kernel Gram matrix between two feature tensors."""
        a_flat = a.view(-1, a.size(-1))
        b_flat = b.view(-1, b.size(-1))
        return classical_kernel_matrix(a_flat, b_flat, gamma=self.kernel.ansatz.gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, input_dim)``.
        Returns
        -------
        torch.Tensor
            Log‑softmaxed tag logits of shape ``(batch, seq_len, tagset_size)``.
        """
        # Encode each token in the sequence
        encoded = self.encode(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), -1)
        # Run through LSTM
        lstm_out, _ = self.lstm(encoded)
        logits = self.hidden2tag(lstm_out)
        return torch.log_softmax(logits, dim=-1)

__all__ = [
    "HybridKernelAutoLSTM",
    "classical_kernel_matrix",
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "ClassicalKernel",
    "ClassicalKernelAnsatz",
    "ClassicalQLSTM",
    "LSTMTagger",
]
