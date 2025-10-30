"""Enhanced classical LSTM with optional quantum‑depth emulation.

The gate functions remain classical, but the linear projections can be
repeated `depth` times to emulate the effect of multiple quantum layers.
When ``n_qubits > 0`` the hidden dimension is projected to `n_qubits`
before the depth‑loop and then projected back to ``hidden_dim``.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLSTM(nn.Module):
    """Drop‑in replacement for a quantum LSTM."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = max(1, depth)

        # Linear projections to the qubit dimension
        self.input_to_q = nn.Linear(
            input_dim + hidden_dim, n_qubits if n_qubits > 0 else hidden_dim
        )
        self.forget_to_q = nn.Linear(
            input_dim + hidden_dim, n_qubits if n_qubits > 0 else hidden_dim
        )
        self.update_to_q = nn.Linear(
            input_dim + hidden_dim, n_qubits if n_qubits > 0 else hidden_dim
        )
        self.output_to_q = nn.Linear(
            input_dim + hidden_dim, n_qubits if n_qubits > 0 else hidden_dim
        )

        # Depth‑loop layers (optional)
        self.depth_layers = nn.ModuleList(
            [
                nn.Linear(
                    n_qubits if n_qubits > 0 else hidden_dim,
                    n_qubits if n_qubits > 0 else hidden_dim,
                )
                for _ in range(self.depth)
            ]
        )

        # Projection back to hidden dimension
        if n_qubits > 0:
            self.proj_back = nn.Linear(n_qubits, hidden_dim)
        else:
            self.proj_back = nn.Identity()

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

    def _depth_loop(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.depth_layers:
            x = F.relu(layer(x))
        return x

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_to_q(combined))
            i = torch.sigmoid(self.input_to_q(combined))
            g = torch.tanh(self.update_to_q(combined))
            o = torch.sigmoid(self.output_to_q(combined))

            # optional depth emulation
            if self.n_qubits > 0:
                f = self._depth_loop(f)
                i = self._depth_loop(i)
                g = self._depth_loop(g)
                o = self._depth_loop(o)

                f = self.proj_back(f)
                i = self.proj_back(i)
                g = self.proj_back(g)
                o = self.proj_back(o)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        return torch.cat(outputs, dim=0), (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum‑depth LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim, hidden_dim, n_qubits=n_qubits, depth=depth
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
