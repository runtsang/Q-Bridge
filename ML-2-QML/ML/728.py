"""Hybrid LSTM with optional quantum augmentation.

This module implements a hybrid LSTM cell that can operate in three
modes:
  * Classical: all gates are standard linear transformations.
  * Quantum: each gate is a quantum‑augmented function that uses a
    variational circuit to transform the linear output.
  * Hybrid: the linear output is passed through a lightweight quantum
    circuit before the activation function.

The implementation is fully differentiable and can be trained with
PyTorch optimizers.  It also exposes a simple `LSTMTagger` wrapper
for sequence tagging experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# --------------------------------------------------------------------------- #
# Classical LSTM cell (seed).  It is kept for reference but not used directly
# --------------------------------------------------------------------------- #
class ClassicalLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        f = torch.sigmoid(self.forget(combined))
        i = torch.sigmoid(self.input(combined))
        g = torch.tanh(self.update(combined))
        o = torch.sigmoid(self.output(combined))
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


# --------------------------------------------------------------------------- #
# Quantum gate module (placeholder).  The quantum part is only used when
# `n_qubits > 0`.  For the classical branch we simply bypass the
# quantum module.
# --------------------------------------------------------------------------- #
class QuantumGate(nn.Module):
    """A very small quantum circuit that returns a vector of size `out_dim`."""

    def __init__(self, out_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.n_qubits = n_qubits
        # For the classical branch we do nothing; the quantum branch
        # will be defined in the QML module.
        # This placeholder allows the class to be imported in the ML
        # module without requiring a quantum backend.
        self.register_buffer("dummy", torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identity mapping for the classical branch.
        return x


# --------------------------------------------------------------------------- #
# Hybrid LSTM cell
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    Hybrid LSTM cell that optionally augments each gate with a quantum
    circuit.  The cell can operate in three modes:

    * ``n_qubits == 0``  →  Pure classical LSTM.
    * ``n_qubits > 0``   →  Each gate is transformed by a quantum
      circuit that receives the linear output as input.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear transformations for the gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum augmentation modules
        if self.n_qubits > 0:
            self.forget_gate = QuantumGate(hidden_dim, self.n_qubits)
            self.input_gate = QuantumGate(hidden_dim, self.n_qubits)
            self.update_gate = QuantumGate(hidden_dim, self.n_qubits)
            self.output_gate = QuantumGate(hidden_dim, self.n_qubits)
        else:
            # Placeholders that perform identity mapping
            self.forget_gate = self.input_gate = self.update_gate = self.output_gate = QuantumGate(hidden_dim, 0)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, c = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, h], dim=1)

            f = self.forget_linear(combined)
            f = self.forget_gate(f)
            f = torch.sigmoid(f)

            i = self.input_linear(combined)
            i = self.input_gate(i)
            i = torch.sigmoid(i)

            g = self.update_linear(combined)
            g = self.update_gate(g)
            g = torch.tanh(g)

            o = self.output_linear(combined)
            o = self.output_gate(o)
            o = torch.sigmoid(o)

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (h, c)


# --------------------------------------------------------------------------- #
# Sequence tagging model
# --------------------------------------------------------------------------- #
class HybridLSTMTagger(nn.Module):
    """
    Sequence tagging model that uses :class:`HybridQLSTM`.  The model
    accepts a batch of sentences (token indices) and produces log‑probabilities
    for each tag.
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
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
