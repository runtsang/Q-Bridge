"""Hybrid LSTM implementation combining classical gates and self‑attention.

The module defines :class:`HybridQLSTM`, a drop‑in replacement for the
original :class:`QLSTM`.  The gates are still classical linear layers,
but an optional self‑attention block (implemented with PyTorch) can be
inserted to let the cell learn to focus on salient input‑hidden
interactions.  The design mirrors the quantum interface so the same
training code can be reused with the quantum version.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalSelfAttention:
    """Light‑weight self‑attention block that mimics the Qiskit interface."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.scale = np.sqrt(self.embed_dim)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # Simple linear projections followed by soft‑max attention
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / self.scale, dim=-1)
        return (scores @ value).numpy()

class HybridQLSTM(nn.Module):
    """Classical LSTM cell with optional self‑attention gating."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_attention: bool = False,
        attention_dim: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_attention = use_attention

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        if self.use_attention:
            self.attention = ClassicalSelfAttention(embed_dim=attention_dim)
            # Learnable projection for attention inputs
            self.att_proj = nn.Linear(input_dim + hidden_dim, attention_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            if self.use_attention:
                # Produce dummy params from a linear map for demo purposes
                rot_params = self.att_proj(combined).detach().cpu().numpy()
                ent_params = self.att_proj(combined).detach().cpu().numpy()
                att_out = self.attention.run(rot_params, ent_params, combined.cpu().numpy())
                # Broadcast attention output to match hidden dimension
                att_tensor = torch.tensor(att_out, device=combined.device).float()
                att_tensor = att_tensor[: self.hidden_dim]
                # Modulate gates with attention output
                f = torch.sigmoid(
                    self.forget_linear(combined) + att_tensor
                )
                i = torch.sigmoid(
                    self.input_linear(combined) + att_tensor
                )
                g = torch.tanh(
                    self.update_linear(combined) + att_tensor
                )
                o = torch.sigmoid(
                    self.output_linear(combined) + att_tensor
                )
            else:
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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

class LSTMTagger(nn.Module):
    """Sequence tagging model that can use the hybrid LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_attention: bool = False,
        attention_dim: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            use_attention=use_attention,
            attention_dim=attention_dim,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
