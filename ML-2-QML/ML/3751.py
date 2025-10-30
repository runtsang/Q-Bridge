"""Hybrid LSTM‑Autoencoder combining classical and quantum modules.

This module defines the classical components of a hybrid LSTM‑autoencoder.
It imports the quantum‑enhanced LSTM implementation from the sibling
`QLSTM_qml` module.  The public class `QLSTM_AE` can be instantiated
with a flag `n_qubits`.  If `n_qubits > 0` a quantum‑enhanced LSTM
(`QLSTM` from qml_code) is used; otherwise the standard `nn.LSTM`
is employed.  The auto‑encoder is a lightweight MLP that shares its
latent dimension with the LSTM hidden state.

The class exposes a `forward` method that returns the tag logits and
the reconstructed input from the auto‑encoder, allowing end‑to‑end
joint training.  The `train_qlstm_ae` helper can be used to train
the entire model on any dataset that yields `(inputs, tags)` pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Guarded import of the quantum LSTM implementation.
try:
    from.QLSTM_qml import QLSTM  # type: ignore
except Exception:  # pragma: no cover
    QLSTM = None  # type: ignore


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
#   Classical auto‑encoder
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    """Configuration for the MLP auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron auto‑encoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(cfg)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
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


# --------------------------------------------------------------------------- #
#   LSTM tagger (classical or hybrid quantum)
# --------------------------------------------------------------------------- #

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between a classical LSTM
    and a quantum‑enhanced LSTM.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the word embeddings.
    hidden_dim : int
        Size of the LSTM hidden state.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of output tags.
    n_qubits : int, optional
        If >0 a quantum LSTM with `n_qubits` qubits is used.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0 and QLSTM is not None:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        # LSTM expects input shape (seq_len, batch, input_size)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


# --------------------------------------------------------------------------- #
#   Hybrid LSTM‑AutoEncoder model
# --------------------------------------------------------------------------- #

class QLSTM_AE(nn.Module):
    """
    Hybrid model that couples a (possibly quantum‑enhanced) LSTM with
    a shared‑latent auto‑encoder.  The auto‑encoder latent dimension is
    tied to the LSTM hidden dimension so that both components can be
    trained jointly.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the word embeddings.
    hidden_dim : int
        Size of the hidden state / latent space.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of output tags.
    n_qubits : int, optional
        If >0 a quantum LSTM is used.
    ae_dropout : float, optional
        Dropout inside the auto‑encoder.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        ae_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tagger = LSTMTagger(
            embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits=n_qubits
        )
        # Auto‑encoder with shared latent dimension = hidden_dim
        ae_cfg = AutoencoderConfig(
            input_dim=hidden_dim,
            latent_dim=hidden_dim,
            hidden_dims=(hidden_dim // 2, hidden_dim // 2),
            dropout=ae_dropout,
        )
        self.autoencoder = AutoencoderNet(ae_cfg)

    def forward(self, sentence: torch.Tensor, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        sentence : torch.Tensor
            Sequence of word indices (shape: seq_len).
        inputs : torch.Tensor
            The original continuous input that will be reconstructed
            by the auto‑encoder (shape: batch, input_dim).

        Returns
        -------
        log_probs : torch.Tensor
            Log‑probabilities of tags (seq_len, tagset_size).
        reconstruction : torch.Tensor
            Reconstructed continuous input (batch, input_dim).
        """
        # 1. Tag prediction
        log_probs = self.tagger(sentence)

        # 2. Latent representation from the last hidden state of the LSTM
        embeds = self.tagger.word_embeddings(sentence)
        lstm_out, (h_n, _) = self.tagger.lstm(
            embeds.view(len(sentence), 1, -1)
        )
        latent = h_n.squeeze(0)  # (hidden_dim)

        # 3. Auto‑encoder reconstruction
        # Expand latent to match batch size of inputs
        latent_expanded = latent.unsqueeze(0).expand(inputs.shape[0], -1)
        reconstruction = self.autoencoder(latent_expanded)
        return log_probs, reconstruction

    def training_step(self, batch, optimizer):
        """
        A single training step for the hybrid model.  The batch should
        contain ``inputs`` (continuous data) and ``tags`` (integer tags).
        """
        inputs, tags = batch
        log_probs, reconstruction = self.forward(tags, inputs)
        tag_loss = F.nll_loss(log_probs, tags)
        ae_loss = F.mse_loss(reconstruction, inputs)
        loss = tag_loss + ae_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item()


# --------------------------------------------------------------------------- #
#   Helper to train the hybrid model
# --------------------------------------------------------------------------- #

def train_qlstm_ae(
    model: QLSTM_AE,
    data_loader: DataLoader,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> list[float]:
    """Train the hybrid LSTM‑auto‑encoder.

    Parameters
    ----------
    model : QLSTM_AE
        The hybrid model.
    data_loader : DataLoader
        Iterable yielding tuples ``(inputs, tags)``.
    epochs : int, optional
        Number of training epochs.
    lr : float, optional
        Optimiser learning rate.
    device : torch.device, optional
        Device to train on.

    Returns
    -------
    history : list[float]
        Loss values per epoch.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            batch = tuple(t.to(device) for t in batch)
            loss = model.training_step(batch, optimizer)
            epoch_loss += loss
        epoch_loss /= len(data_loader)
        history.append(epoch_loss)
    return history


__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "train_autoencoder",
    "LSTMTagger",
    "QLSTM_AE",
    "train_qlstm_ae",
]
