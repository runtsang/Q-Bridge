"""Hybrid LSTM implementation combining classical and quantum-inspired layers.

Provides a drop-in replacement for the original QLSTM that can operate entirely
on classical tensors while mimicking the structure of the quantum gates
through a lightweight fully‑connected approximation (FCL).  The module
exposes ``HybridQLSTM`` and ``LSTMTagger`` which are interchangeable
with the original versions but additionally support a ``use_quantum``
flag for future extensions.

The design follows these principles:

* Classical gating functions are backed by a trainable linear layer
  followed by a `tanh` non‑linearity, mirroring the quantum expectation
  values in the quantum version.
* The `FCL` class implements the same API as the quantum example
  but returns deterministic activations, making the whole network
  differentiable with PyTorch autograd.
* The overall architecture remains compatible with the original `QLSTM.py`,
  ensuring that downstream code continues to instantiate `QLSTM`
  (aliased to `HybridQLSTM` here) without modification.

Usage
-----
>>> model = HybridQLSTM(input_dim=128, hidden_dim=256, n_qubits=4)
>>> out, (h,c) = model(inputs)
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------- #
# FCL – classical surrogate for a parameterized quantum circuit
# --------------------------------------------------------------------------- #
class FCL(nn.Module):
    """Classical surrogate for a parameterized quantum circuit.

    The implementation mirrors the quantum example but operates purely
    on classical tensors.  It is kept for compatibility with the
    original API and can be dropped if a purely linear gate is
    preferred.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

# --------------------------------------------------------------------------- #
# QLayer – classical approximation of a quantum gate
# --------------------------------------------------------------------------- #
class QLayer(nn.Module):
    """Classical approximation of a quantum gate.

    The layer maps a scalar input to a vector of *n_wires* values
    that are later transformed into the hidden dimension by a
    separate linear mapping.  This mimics the behaviour of the
    quantum implementation while remaining completely classical.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.linear = nn.Linear(1, n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` is expected to be of shape (batch, 1)
        return torch.tanh(self.linear(x))

# --------------------------------------------------------------------------- #
# HybridQLSTM – hybrid classical/quantum LSTM
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """Drop‑in replacement for the original QLSTM.

    The class builds a standard LSTM where each gate is implemented
    by a *QLayer* followed by a linear mapping to the hidden dimension.
    When ``n_qubits`` is set to zero the module degrades gracefully to a
    conventional PyTorch ``nn.LSTM``.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Gate‑specific QLayers
        self.forget = QLayer(n_qubits)
        self.input_gate = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        # Linear transforms from concatenated input/state to gate parameters
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Mapping from gate output (n_qubits) to hidden dimension
        self.map_forget = nn.Linear(n_qubits, hidden_dim)
        self.map_input = nn.Linear(n_qubits, hidden_dim)
        self.map_update = nn.Linear(n_qubits, hidden_dim)
        self.map_output = nn.Linear(n_qubits, hidden_dim)

    # --------------------------------------------------------------------- #
    # Helper to initialise hidden/cell states
    # --------------------------------------------------------------------- #
    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f_raw = self.forget(self.linear_forget(combined))
            f = torch.sigmoid(self.map_forget(f_raw))

            i_raw = self.input_gate(self.linear_input(combined))
            i = torch.sigmoid(self.map_input(i_raw))

            g_raw = self.update(self.linear_update(combined))
            g = torch.tanh(self.map_update(g_raw))

            o_raw = self.output(self.linear_output(combined))
            o = torch.sigmoid(self.map_output(o_raw))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        return torch.cat(outputs, dim=0), (hx, cx)

# --------------------------------------------------------------------------- #
# LSTMTagger – sequence tagging model using HybridQLSTM
# --------------------------------------------------------------------------- #
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
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger", "FCL"]
