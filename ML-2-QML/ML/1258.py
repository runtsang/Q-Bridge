"""Classical LSTM with a generative decoder for sequence tagging and generation."""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLSTMGen204(nn.Module):
    """A classical LSTM cell that returns the hidden state, a per‑step generative
    probability distribution over a target vocabulary, and optionally applies
    quantum‑inspired attention (disabled in the classical variant)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        n_qubits: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_qubits = n_qubits

        # LSTM gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Generative decoder
        self.decoder = nn.Linear(hidden_dim, vocab_size)

        # Optional quantum attention flag (not used in classical variant)
        self.use_quantum_attention = n_qubits > 0

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass that returns:
        - hidden states (for tagging)
        - hidden/cell states
        - generative logits over the target vocabulary
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []
        gen_outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
            gen_outputs.append(self.decoder(hx).unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        gen_outputs = torch.cat(gen_outputs, dim=0)
        return outputs, (hx, cx), gen_outputs

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTaggerGen204(nn.Module):
    """Sequence tagging model that uses QLSTMGen204."""

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
        self.lstm = QLSTMGen204(embedding_dim, hidden_dim, vocab_size, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds = self.word_embeddings(sentence)
        lstm_out, _, gen_out = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_log_probs = F.log_softmax(tag_logits, dim=1)
        gen_log_probs = F.log_softmax(gen_out, dim=2)
        return tag_log_probs, gen_log_probs


__all__ = ["QLSTMGen204", "LSTMTaggerGen204"]
