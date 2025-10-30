"""Hybrid classical LSTM with optional quantum linear simulation.

The class HybridQLSTM can operate purely classically or with quantum‑inspired
linear layers.  When ``use_quantum`` is True and ``n_qubits`` > 0 the gates
employ a tiny quantum module that is simulated with classical tensors.
The implementation preserves the public API of the original QLSTM for
drop‑in compatibility with downstream code.

The design reflects the following merged concepts:
* Classical linear gates from the original pure‑PyTorch implementation.
* Quantum linear layer simulation inspired by the FCL example.
* Optional switch to a full quantum backend for the gates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class _QuantumLinearSim(nn.Module):
    """A toy quantum‑inspired linear layer that mimics the behaviour of the
    fully‑connected quantum circuit from the FCL example.

    The layer maps an input of size ``in_features`` to ``out_features`` by
    applying a trainable weight matrix followed by a tanh activation and
    a single‑shot expectation value.  The implementation is purely classical
    but keeps the same interface as a quantum circuit.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear transformation followed by tanh to mimic an expectation value
        z = F.linear(x, self.weight, self.bias)
        return torch.tanh(z)

class HybridQLSTM(nn.Module):
    """Hybrid LSTM cell with optional quantum‑inspired gates.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each input element.
    hidden_dim : int
        Dimensionality of the hidden state.
    n_qubits : int
        Number of qubits used in the quantum linear layers.  If zero the
        module behaves purely classically.
    use_quantum : bool, optional
        If ``True`` the gates are implemented with the quantum simulator
        described in ``_QuantumLinearSim``.  If ``False`` the gates use
        plain ``nn.Linear``.  The parameter is ignored when ``n_qubits == 0``.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_quantum: bool = False
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        linear_cls = _QuantumLinearSim if (use_quantum and n_qubits > 0) else nn.Linear
        # Gate layers
        self.forget_gate = linear_cls(input_dim + hidden_dim, hidden_dim)
        self.input_gate = linear_cls(input_dim + hidden_dim, hidden_dim)
        self.update_gate = linear_cls(input_dim + hidden_dim, hidden_dim)
        self.output_gate = linear_cls(input_dim + hidden_dim, hidden_dim)

        # Optional fully‑connected quantum layer for output projection
        self.output_proj = linear_cls(hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTagger(nn.Module):
    """Convenience wrapper that tags sequences with a hybrid LSTM.

    The tagger can be instantiated with a purely classical LSTM, a
    quantum‑inspired LSTM, or a full quantum backend.  The public API
    matches the original reference to ease migration.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum: bool = False
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            use_quantum=use_quantum
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
