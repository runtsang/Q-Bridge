"""
ML implementation of a hybrid LSTM with quantum‑gate emulation.

The module defines a class `QLSTM` that mirrors the interface of the quantum
counterpart but is fully classical.  A shared linear projection is followed
by a small feed‑forward network that mimics a quantum circuit.  The
architecture is configurable through `depth`, `share_gates`, and
`quantum_gate_only`.  The `LSTMTagger` uses either this class or the
standard `nn.LSTM`.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLSTM(nn.Module):
    """Hybrid LSTM where each gate is a small classical network that mimics a
    quantum circuit.  The design is intentionally modular so that the same
    class can be swapped with the quantum implementation without changing
    downstream code.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        projection_dim: int = 64,
        *,
        depth: int = 1,
        return_all: bool = False,
        share_gates: bool = False,
        quantum_gate_only: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.projection_dim = projection_dim
        self.depth = depth
        self.return_all = return_all
        self.share_gates = share_gates
        self.quantum_gate_only = quantum_gate_only

        # Shared projection for all gates
        self.shared_proj = nn.Linear(input_dim + hidden_dim, projection_dim)

        # Linear map from projection to qubit space
        self.linear_to_qubits = nn.Linear(projection_dim, n_qubits, bias=False)

        # Gate emulators
        if share_gates:
            gate = nn.ModuleList([self._make_gate() for _ in range(depth)])
            self.forget_gate = self.input_gate = self.update_gate = self.output_gate = gate
        else:
            self.forget_gate = nn.ModuleList([self._make_gate() for _ in range(depth)])
            self.input_gate = nn.ModuleList([self._make_gate() for _ in range(depth)])
            self.update_gate = nn.ModuleList([self._make_gate() for _ in range(depth)])
            self.output_gate = nn.ModuleList([self._make_gate() for _ in range(depth)])

        # Map qubit outputs to hidden dimension
        self.linear_qubit_to_hidden = nn.Linear(n_qubits, hidden_dim, bias=False)

    def _make_gate(self) -> nn.Module:
        """A tiny feed‑forward network that acts as a quantum‑style gate."""
        return nn.Sequential(
            nn.Linear(self.n_qubits, self.n_qubits),
            nn.Sigmoid(),
        )

    def _apply_gate(self, gate_list: nn.ModuleList, vec: torch.Tensor) -> torch.Tensor:
        for gate in gate_list:
            vec = gate(vec)
        return vec

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

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            proj = self.shared_proj(combined)
            qinput = combined if self.quantum_gate_only else self.linear_to_qubits(proj)
            f = self._apply_gate(self.forget_gate, qinput)
            i = self._apply_gate(self.input_gate, qinput)
            g = self._apply_gate(self.update_gate, qinput)
            o = self._apply_gate(self.output_gate, qinput)
            f = torch.sigmoid(self.linear_qubit_to_hidden(f))
            i = torch.sigmoid(self.linear_qubit_to_hidden(i))
            g = torch.tanh(self.linear_qubit_to_hidden(g))
            o = torch.sigmoid(self.linear_qubit_to_hidden(o))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        if self.return_all:
            return stacked, (hx, cx)
        return stacked[-1], (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence‑tagging model that can switch between the hybrid LSTM and
    a standard `nn.LSTM`.  The API is identical to the original seed.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits,
                **kwargs,
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
