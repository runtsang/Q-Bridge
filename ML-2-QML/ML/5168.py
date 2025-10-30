"""Hybrid LSTM Tagger with optional quantum components.

This module defines a classical PyTorch model that can optionally
delegate its recurrent, sampling and classification sub‑modules to
quantum implementations.  The quantum sub‑modules are imported lazily
so that the model can run on a plain CPU/GPU without any quantum
backend.

The design is inspired by the following reference pairs:

1. QLSTM: classical vs quantum LSTM cells.
2. SamplerQNN: a small neural sampler that mimics a quantum sampler.
3. QuantumClassifierModel: a quantum classifier circuit with a classical
   analogue.
4. FraudDetection: photonic layers that apply linear scaling and
   bias‑shifting – implemented here as a post‑activation
   ``ScaleShift`` module.

The model offers a single ``forward`` method that accepts a sequence
of token indices and returns log‑probabilities over the tag set.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------- #
#  Classical helpers
# --------------------------------------------------------------------------- #

class ScaleShift(nn.Module):
    """
    Applies a linear scaling and bias shift after an activation.
    Mimics the post‑activation processing seen in the photonic fraud
    detection example.
    """
    def __init__(self, scale: float = 1.0, shift: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))
        self.register_buffer("shift", torch.tensor(shift))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift

class SamplerQNN(nn.Module):
    """
    Classical sampler network that produces a probability vector for
    each hidden state.  The architecture mirrors the simple
    ``SamplerQNN`` from the reference pair – a 2‑layer feed‑forward
    network ending in a softmax.
    """
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class QClassifier(nn.Module):
    """
    Classical classifier that imitates the quantum classifier
    circuit.  It is a simple feed‑forward network that maps the
    LSTM hidden state to a two‑class logit vector.
    """
    def __init__(self, hidden_dim: int, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = hidden_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------------- #
#  Hybrid Tagger
# --------------------------------------------------------------------------- #

class HybridQLSTM(nn.Module):
    """
    Sequence‑tagging model that can operate in purely classical mode
    or delegate recurrent, sampling and classification to quantum
    sub‑modules.

    Parameters
    ----------
    embedding_dim : int
        Size of the token embeddings.
    hidden_dim : int
        Size of the LSTM hidden state.
    vocab_size : int
        Vocabulary size.
    tagset_size : int
        Number of tags.
    n_qubits : int, optional
        If > 0 the model uses a quantum LSTM cell.  Otherwise a
        classical ``nn.LSTM`` is used.
    use_sampler : bool, optional
        Whether to apply a sampler to the hidden state before
        classification.
    use_classifier : bool, optional
        Whether to replace the linear tag head with a quantum
        classifier.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_sampler: bool = False,
        use_classifier: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.n_qubits = n_qubits
        self.use_sampler = use_sampler
        self.use_classifier = use_classifier

        # Recurrent core – classical or quantum
        if n_qubits > 0:
            # Lazy import to avoid importing torchquantum unless needed.
            from. import HybridQLSTM as QuantumLSTM
            self.lstm = QuantumLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Tag head – linear or quantum classifier
        if use_classifier:
            # Lazy import to keep the module pure classical until needed.
            from. import QClassifierQuantum as QuantumClassifier
            self.tag_head = QuantumClassifier(hidden_dim)
        else:
            self.tag_head = nn.Linear(hidden_dim, tagset_size)

        # Optional sampler
        if use_sampler:
            self.sampler = SamplerQNN(hidden_dim, hidden_dim)
        else:
            self.sampler = None

        # Post‑activation scale/shift (FraudDetection style)
        self.scale_shift = ScaleShift(scale=1.0, shift=0.0)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Input token indices of shape (seq_len, batch).

        Returns
        -------
        torch.Tensor
            Log‑probabilities over the tag set of shape (seq_len, batch, tagset_size).
        """
        embeds = self.embedding(sentence)  # (seq_len, batch, embed)
        lstm_out, _ = self.lstm(embeds)
        hidden = lstm_out  # (seq_len, batch, hidden_dim)

        # Optional sampling – element‑wise multiplication with probabilities
        if self.sampler is not None:
            probs = self.sampler(hidden.view(-1, hidden.size(-1)))
            probs = probs.view_as(hidden)
            hidden = hidden * probs

        # Tag head
        logits = self.tag_head(hidden.view(-1, hidden.size(-1)))

        # Apply fraud‑style scaling/shift
        logits = self.scale_shift(logits)

        return F.log_softmax(logits, dim=-1).view(sentence.size(0), -1, logits.size(-1))

    def _init_states(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Utility to initialise hidden/cell states for the quantum LSTM.
        """
        if hasattr(self.lstm, "_init_states"):
            return self.lstm._init_states(batch_size, device)
        else:
            return torch.zeros(batch_size, self.lstm.hidden_size, device=device), \
                   torch.zeros(batch_size, self.lstm.hidden_size, device=device)

__all__ = ["HybridQLSTM"]
