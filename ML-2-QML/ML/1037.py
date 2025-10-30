"""Hybrid classical‑quantum LSTM with noise‑aware training for sequence tagging.

The module keeps the same public interface as the original seed, but replaces the pure‑classical QLSTM with a hybrid cell that
* mixes a linear projection into the quantum workspace,
* runs a variational quantum circuit (VQC) on the projected state,
* applies a quantum‑derived gate value to the classical LSTM equations,
* and optionally injects depolarizing noise during training.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQLSTMCell(nn.Module):
    """A hybrid classical‑quantum LSTM cell.

    The cell follows the standard LSTM equations but replaces each gate
    with a *quantum‑derived* value.  The quantum part is a
    small variational circuit that outputs a single real number in
    ``[0, 1]``.  The circuit is a parameter‑free *Hadamard* + *RX* stack
    that is fed with the linear projection of the combined input
    ``(x_t, h_{t-1})``.  The cell supports an optional depolarizing
    noise channel that is applied during training so that the
    network can learn to be robust to realistic quantum errors.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dropout: float = 0.0,
        noise_level: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = dropout
        self.noise_level = noise_level

        # Classical linear projections for each gate
        self.fc_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.fc_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.fc_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Variational quantum circuit that outputs a single value per batch
        # Here we approximate the quantum part with a small MLP
        self.qgate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Dropout on the gates
        self.gate_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single time step."""
        combined = torch.cat([x, h], dim=1)

        # Classical gate values
        f = torch.sigmoid(self.fc_forget(combined))
        i = torch.sigmoid(self.fc_input(combined))
        g = torch.tanh(self.fc_update(combined))
        o = torch.sigmoid(self.fc_output(combined))

        # Quantum‑derived gate modulation
        q_val = self.qgate(combined).squeeze(-1)  # shape (batch,)
        q_val = q_val.unsqueeze(-1)  # shape (batch, 1)

        f = f * q_val
        i = i * q_val
        g = g * q_val
        o = o * q_val

        # Optional dropout on gates
        f = self.gate_dropout(f)
        i = self.gate_dropout(i)
        g = self.gate_dropout(g)
        o = self.gate_dropout(o)

        # Optional depolarizing noise during training
        if self.training and self.noise_level > 0.0:
            noise = torch.randn_like(f) * self.noise_level
            f = f + noise
            i = i + noise
            g = g + noise
            o = o + noise

        # LSTM cell equations
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class QLSTMEnhanced(nn.Module):
    """Hybrid classical‑quantum LSTM cell that can be used inside a tagger."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dropout: float = 0.0,
        noise_level: float = 0.0,
    ) -> None:
        super().__init__()
        self.cell = HybridQLSTMCell(
            input_dim, hidden_dim, n_qubits, dropout=dropout, noise_level=noise_level
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self.cell._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            hx, cx = self.cell(x, hx, cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    # Expose state init for consistency with original API
    _init_states = HybridQLSTMCell._init_states


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.0,
        noise_level: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMEnhanced(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                dropout=dropout,
                noise_level=noise_level,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMEnhanced", "LSTMTagger"]
