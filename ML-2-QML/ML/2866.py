"""Hybrid classical LSTMTagger with optional self‑attention.

This module defines a QLSTMTagger that can operate in a purely classical mode
or enable a lightweight self‑attention layer.  The class names and signatures
match the original QLSTM.py so that existing training scripts can import and
use it without modification.  The attention implementation is a NumPy‑based
soft‑max attention that mirrors the interface of the quantum version.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalSelfAttention:
    """Simple attention mechanism compatible with the quantum interface."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a self‑attention weighted sum.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters used to generate the query vectors.
        entangle_params : np.ndarray
            Parameters used to generate the key vectors.
        inputs : np.ndarray
            Input matrix of shape (seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Attention‑weighted values of shape (seq_len, embed_dim).
        """
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class QLSTM(nn.Module):
    """Classical LSTM cell using linear gates."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class QLSTMTagger(nn.Module):
    """Sequence tagging model that can optionally apply self‑attention."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_attention: bool = False,
        attention_type: str = "classical",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Optional attention
        self.use_attention = use_attention
        if use_attention:
            if attention_type == "classical":
                self.attention = ClassicalSelfAttention(embed_dim=embedding_dim)
            else:
                raise ValueError("Only classical attention is available in the ML variant")

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        if self.use_attention:
            # Convert to NumPy for the attention routine
            attn_input = embeds.detach().cpu().numpy()
            # Dummy rotation/entangle params (identity)
            rot = np.eye(self.hidden_dim).flatten()
            ent = np.eye(self.hidden_dim).flatten()
            attn_output = self.attention.run(rot, ent, attn_input)
            embeds = torch.as_tensor(attn_output, device=embeds.device, dtype=embeds.dtype)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMTagger", "QLSTM"]
