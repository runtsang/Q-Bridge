"""Hybrid classical LSTMTagger with optional autoencoder and fidelity‑based pruning.

The module mirrors the original :class:`QLSTM` API but adds:
* an optional autoencoder that compresses hidden states before classification,
* a graph‑based fidelity pruning that can reset hidden states when similarity falls below a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

Tensor = torch.Tensor


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
# Autoencoder utilities
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
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

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def Autoencoder(cfg: AutoencoderConfig) -> AutoencoderNet:
    """Factory that returns a configured autoencoder."""
    return AutoencoderNet(cfg)


def train_autoencoder(
    model: AutoencoderNet,
    data: Tensor,
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
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
# Fidelity‑based graph utilities
# --------------------------------------------------------------------------- #
def _cosine_similarity(a: Tensor, b: Tensor) -> float:
    """Cosine similarity between two vectors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()


def build_fidelity_graph(
    hidden_states: Tensor, threshold: float, *, secondary: Optional[float] = None, secondary_weight: float = 0.5
) -> nx.Graph:
    """
    Build a weighted graph where nodes are timestep indices and edges are added
    when the cosine similarity of hidden states exceeds ``threshold``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(hidden_states.size(0)))
    for i in range(hidden_states.size(0)):
        for j in range(i + 1, hidden_states.size(0)):
            fid = _cosine_similarity(hidden_states[i], hidden_states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Hybrid LSTMTagger
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    Classical LSTMTagger extended with:
    * Optional autoencoder that compresses hidden states before classification.
    * Optional fidelity‑based pruning that can reset hidden states when similarity
      falls below ``fidelity_threshold``.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        *,
        autoencoder_cfg: Optional[AutoencoderConfig] = None,
        fidelity_threshold: float = 0.0,
        use_fidelity: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Classical LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Optional autoencoder for hidden state compression
        self.autoencoder: Optional[AutoencoderNet] = None
        if autoencoder_cfg is not None:
            autoencoder_cfg.input_dim = hidden_dim
            self.autoencoder = Autoencoder(autoencoder_cfg)
            self.hidden2tag = nn.Linear(autoencoder_cfg.latent_dim, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.fidelity_threshold = fidelity_threshold
        self.use_fidelity = use_fidelity

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : LongTensor
            Sequence of token indices with shape (batch, seq_len).

        Returns
        -------
        log_probs : Tensor
            Log‑probabilities over the tagset, shape (batch, seq_len, tagset_size).
        """
        embeds = self.word_embeddings(sentence)  # (batch, seq_len, embed)
        lstm_out, _ = self.lstm(embeds)  # (batch, seq_len, hidden)

        # Optional fidelity‑based pruning
        if self.use_fidelity and self.fidelity_threshold > 0.0:
            # Compute pairwise cosine similarity between consecutive hidden states
            similarities = torch.cdist(lstm_out, lstm_out, p=2).diagonal(offset=1)
            mask = similarities >= self.fidelity_threshold
            # Zero out hidden states that do not satisfy threshold
            lstm_out = lstm_out * mask.unsqueeze(-1).float()

        # Optional autoencoder compression
        if self.autoencoder is not None:
            latent = self.autoencoder.encode(lstm_out)  # (batch, seq_len, latent_dim)
            logits = self.hidden2tag(latent)
        else:
            logits = self.hidden2tag(lstm_out)

        return F.log_softmax(logits, dim=-1)

    def get_fidelity_graph(self, hidden_states: Tensor) -> nx.Graph:
        """
        Compute the fidelity graph of the provided hidden states.
        """
        return build_fidelity_graph(hidden_states, self.fidelity_threshold)

__all__ = ["HybridQLSTM", "AutoencoderConfig", "AutoencoderNet", "Autoencoder", "train_autoencoder"]
