"""Hybrid classical LSTM tagger with optional auto‑encoder preprocessor.

The module extends the original QLSTM implementation by adding a classical
auto‑encoder that can be used as a lightweight dimensionality reducer before
the LSTM.  The LSTM itself can switch between a pure PyTorch implementation
or a quantum‑enhanced gate variant (the latter is provided by the QML module
and is imported lazily).  This design allows a single API to experiment
with both classical and quantum building blocks while keeping the training
pipelines identical.

Typical usage:

>>> from HybridQLSTM import HybridQLSTM
>>> model = HybridQLSTM(embedding_dim=50, hidden_dim=128, vocab_size=1000,
...                     tagset_size=10, n_qubits=0, ae_type='classical')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

# --------------------------------------------------------------------------- #
# Classical Auto‑Encoder
# --------------------------------------------------------------------------- #

@dataclass
class AEConfig:
    input_dim: int
    latent_dim: int = 16
    hidden_dims: Tuple[int,...] = (64, 32)
    dropout: float = 0.1

class ClassicalAE(nn.Module):
    """A lightweight MLP auto‑encoder inspired by the reference `Autoencoder.py`."""
    def __init__(self, cfg: AEConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: list[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
# Hybrid LSTM Tagger
# --------------------------------------------------------------------------- #

class HybridQLSTM(nn.Module):
    """
    Sequence tagger that optionally pre‑processes word embeddings with a
    classical or quantum auto‑encoder and then runs them through either a
    classical LSTM or a quantum‑gate LSTM.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the input embeddings.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of output tags.
    n_qubits : int, default 0
        If > 0 the LSTM gates are realised by a quantum module.
    ae_type : str, default 'none'
        One of ``'none'``, ``'classical'`` or ``'quantum'``.  The quantum
        variant is available only in the QML module.
    ae_cfg : dict | None
        Configuration dict for the auto‑encoder.  If ``ae_type=='classical'``
        it is passed to :class:`ClassicalAE`.  For the quantum case it is
        forwarded to the quantum auto‑encoder constructor.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        ae_type: str = "none",
        ae_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Auto‑encoder branch
        if ae_type == "classical":
            cfg_dict = ae_cfg or {}
            cfg = AEConfig(
                input_dim=embedding_dim,
                latent_dim=cfg_dict.get("latent_dim", 16),
                hidden_dims=tuple(cfg_dict.get("hidden_dims", (64, 32))),
                dropout=cfg_dict.get("dropout", 0.1),
            )
            self.preproc = ClassicalAE(cfg)
        elif ae_type == "quantum":
            raise NotImplementedError(
                "Quantum auto‑encoder is available only in the QML module."
            )
        else:
            self.preproc = None

        # LSTM branch
        if n_qubits > 0:
            # Import lazily to avoid a hard dependency on qiskit at import time.
            from. import quantum_lstm  # type: ignore
            self.lstm = quantum_lstm.QuantumLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.LongTensor of shape (seq_len,)
            Token indices of a single sentence.

        Returns
        -------
        log_probs : torch.Tensor of shape (seq_len, tagset_size)
            Log‑softmaxed tag logits.
        """
        x = self.word_embeddings(sentence)  # (seq_len, embedding_dim)
        if self.preproc is not None:
            x = self.preproc(x)
        # LSTM expects (seq_len, batch, features)
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQLSTM", "ClassicalAE", "AEConfig"]
