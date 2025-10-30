"""Classical hybrid LSTM with optional dropout and per-gate dropout.

The implementation is purely classical; quantum gates are not supported.
If ``n_qubits > 0`` or any entry in ``gate_mode`` is ``True``, a
``NotImplementedError`` is raised to avoid accidental use of quantum logic.
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQLSTM(nn.Module):
    """Hybrid LSTM cell with optional per-gate dropout.

    The implementation is purely classical; quantum gates are not supported.
    If ``n_qubits > 0`` or any entry in ``gate_mode`` is ``True``, a
    :class:`NotImplementedError` is raised to avoid accidental use of
    quantum logic.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        gate_mode: Optional[Dict[str, bool]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_qubits > 0 or (gate_mode and any(gate_mode.values())):
            raise NotImplementedError(
                "The classical HybridQLSTM does not support quantum gates. "
                "Set ``n_qubits=0`` and all gate_mode entries to ``False``."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Linear projections for all gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)

        # Orthogonal initialization
        for lin in (
            self.forget_linear,
            self.input_linear,
            self.update_linear,
            self.output_linear,
        ):
            nn.init.orthogonal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _dropout(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.dropout > 0.0:
            return F.dropout(tensor, p=self.dropout, training=self.training)
        return tensor

    def _init_states(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_len, batch_size, _ = inputs.shape
        device = inputs.device

        hx, cx = self._init_states(batch_size, device) if states is None else states

        outputs = []

        for t in range(seq_len):
            x_t = inputs[t]
            combined = torch.cat([x_t, hx], dim=1)

            f = torch.sigmoid(self._dropout(self.forget_linear(combined)))
            i = torch.sigmoid(self._dropout(self.input_linear(combined)))
            g = torch.tanh(self._dropout(self.update_linear(combined)))
            o = torch.sigmoid(self._dropout(self.output_linear(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses :class:`HybridQLSTM`."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        gate_mode: Optional[Dict[str, bool]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            gate_mode=gate_mode,
            dropout=dropout,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)


__all__ = ["HybridQLSTM", "LSTMTagger"]
