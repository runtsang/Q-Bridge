"""Hybrid kernel, auto‑encoder, and LSTM tagger.

This module implements the classical side of the unified
`QuantumKernelCombined` interface.  It mirrors the API of the
seeds but adds a small wrapper that can later be swapped for the
quantum implementation in the companion `qml_code` module.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence, List

# --------------------------------------------------------------------------- #
# Classical RBF kernel
# --------------------------------------------------------------------------- #

class ClassicalKernel(nn.Module):
    """Pure‑PyTorch radial‑basis‑function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def gram_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Auto‑encoder
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder mirroring the seed."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
# LSTM tagger
# --------------------------------------------------------------------------- #

class LSTMTagger(nn.Module):
    """Sequence tagger that uses a classical LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return torch.log_softmax(tag_logits, dim=-1)

# --------------------------------------------------------------------------- #
# Composite class
# --------------------------------------------------------------------------- #

class QuantumKernelCombined:
    """
    Unified interface that can operate in a fully classical mode
    or switch to quantum back‑ends when the companion `qml_code`
    module is used.

    Parameters
    ----------
    use_quantum_kernel : bool
        If True, raises ``NotImplementedError`` in this module.
    use_quantum_lstm : bool
        If True, raises ``NotImplementedError`` in this module.
    use_quantum_autoencoder : bool
        If True, raises ``NotImplementedError`` in this module.
    """
    def __init__(self,
                 use_quantum_kernel: bool = False,
                 use_quantum_lstm: bool = False,
                 use_quantum_autoencoder: bool = False,
                 kernel_gamma: float = 1.0,
                 autoencoder_config: AutoencoderConfig | None = None,
                 lstm_params: Tuple[int, int, int, int] | None = None):
        if use_quantum_kernel:
            raise NotImplementedError("Quantum kernel is available only in the quantum module.")
        if use_quantum_lstm:
            raise NotImplementedError("Quantum LSTM is available only in the quantum module.")
        if use_quantum_autoencoder:
            raise NotImplementedError("Quantum auto‑encoder is available only in the quantum module.")

        # Kernel
        self.kernel = ClassicalKernel(gamma=kernel_gamma)

        # Auto‑encoder
        if autoencoder_config is None:
            # Minimal placeholder – user must supply a proper config
            autoencoder_config = AutoencoderConfig(input_dim=0)
        self.autoencoder = AutoencoderNet(autoencoder_config)

        # LSTM tagger
        if lstm_params is None:
            lstm_params = (50, 100, 5000, 10)  # (emb_dim, hid_dim, vocab_sz, tag_sz)
        emb_dim, hid_dim, vocab_sz, tag_sz = lstm_params
        self.tagger = LSTMTagger(emb_dim, hid_dim, vocab_sz, tag_sz)

    # ----------------------------------------------------------------------- #
    # Kernel utilities
    # ----------------------------------------------------------------------- #
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix of the chosen kernel."""
        return self.kernel.gram_matrix(a, b)

    # ----------------------------------------------------------------------- #
    # Auto‑encoder utilities
    # ----------------------------------------------------------------------- #
    def train_autoencoder(self,
                          data: torch.Tensor,
                          epochs: int = 100,
                          batch_size: int = 64,
                          lr: float = 1e-3) -> List[float]:
        """Simple reconstruction training loop."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon = self.autoencoder(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            history.append(epoch_loss / len(dataset))
        return history

    # ----------------------------------------------------------------------- #
    # LSTM utilities
    # ----------------------------------------------------------------------- #
    def tag_sequence(self, sentence: torch.Tensor) -> torch.Tensor:
        """Run the tagger on a single sentence."""
        return self.tagger(sentence)

__all__ = ["QuantumKernelCombined", "ClassicalKernel", "AutoencoderNet", "LSTMTagger"]
