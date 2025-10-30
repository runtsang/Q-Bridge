"""Hybrid LSTM with optional classical self‑attention and quantum‑gate fallback.

The module keeps the original API while adding the following features:
* `ClassicalSelfAttention` mimics the quantum attention interface.
* `use_attention` flag enables a lightweight attention mechanism.
* When `n_qubits=0` the gates are simple linear layers; otherwise
  quantum‑inspired logic is used (see the QML counterpart).
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalSelfAttention:
    """Simple self‑attention that uses the same call signature as the quantum version."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class QLSTM(nn.Module):
    """LSTM cell that can operate with classical linear gates or a quantum‑inspired
    gate implementation.  An optional self‑attention layer can be applied
    after the recurrent computation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_attention = use_attention

        gate_dim = hidden_dim
        if n_qubits > 0:
            # Placeholder for quantum gates – the actual implementation lives in the QML module.
            self.forget = nn.Identity()
            self.input = nn.Identity()
            self.update = nn.Identity()
            self.output = nn.Identity()
            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            self.forget = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.input = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.update = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.output = nn.Linear(input_dim + hidden_dim, gate_dim)

        if use_attention:
            # Parameters for the classical attention mimic the quantum rotation/entangle params.
            self.attn_rot_params = nn.Parameter(torch.randn(hidden_dim))
            self.attn_ent_params = nn.Parameter(torch.randn(hidden_dim - 1))
            self.attention = ClassicalSelfAttention(embed_dim=hidden_dim)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)

        if self.use_attention:
            # Apply classical self‑attention to the sequence of hidden states.
            attn_inputs = outputs.detach().cpu().numpy()
            rot = self.attn_rot_params.detach().cpu().numpy()
            ent = self.attn_ent_params.detach().cpu().numpy()
            attn_out = self.attention.run(rot, ent, attn_inputs)
            attn_weights = torch.from_numpy(attn_out).to(outputs.device)
            attn_weights = F.softmax(attn_weights, dim=0)
            context = torch.sum(outputs * attn_weights.unsqueeze(1), dim=0, keepdim=True)
            outputs = outputs + context

        return outputs, (hx, cx)


class LSTMTagger(nn.Module):
    """Drop‑in replacement for the original tagger that supports the new LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, use_attention=use_attention)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
