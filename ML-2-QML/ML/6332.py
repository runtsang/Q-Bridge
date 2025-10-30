"""Hybrid LSTM: Classical gates + optional quantum gate for forget, with tunable strength."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQLSTM(nn.Module):
    """A classical LSTM cell that optionally injects a quantum‑based forget gate.

    Parameters
    ----------
    input_dim : int
        Dimension of input embeddings.
    hidden_dim : int
        Hidden state size.
    quantum_strength : float, default=0.0
        Weight (0–1) of the quantum contribution in the forget gate.
        When ``0`` the cell reduces to a vanilla LSTM; when ``1`` the gate
        fully relies on quantum outputs.
    n_qubits : int, default=0
        Number of qubits for the quantum circuit.  Must be >0 when
        quantum_strength > 0.

    The cell keeps the same public API as the original QLSTM, so
    ``HybridQLSTM`` can be dropped‑in replace‑able.
    """

    class _QuantumLayer(nn.Module):
        """Mock quantum layer that returns a sigmoid‑based 1‑bit output."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Parameters for a simple parametrized rotation
            self.params = nn.Parameter(torch.randn(n_wires))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # In a real system we would interface with a quantum simulator.
            # This mock‑function returns a sigmoid‑based 1‑bit output.
            return torch.sigmoid(torch.sum(x, dim=1, keepdim=True))

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        quantum_strength: float = 0.0,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.quantum_strength = float(quantum_strength)
        self.n_qubits = n_qubits

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        if self.quantum_strength > 0.0:
            if self.n_qubits <= 0:
                raise ValueError("n_qubits must be > 0 when quantum_strength > 0")
            self.quantum_forget = self._QuantumLayer(self.n_qubits)
            self.quantum_proj = nn.Linear(gate_dim, self.n_qubits)
        else:
            self.quantum_forget = None

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f_class = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            if self.quantum_forget is not None:
                # Project to qubit space
                q_input = torch.tanh(self.quantum_proj(combined))
                q_out = self.quantum_forget(q_input)
                # Combine classical and quantum forget gates
                f = (1.0 - self.quantum_strength) * f_class + self.quantum_strength * q_out
            else:
                f = f_class

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses either :class:`HybridQLSTM` or ``nn.LSTM``."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        quantum_strength: float = 0.0,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if quantum_strength > 0.0:
            self.lstm = HybridQLSTM(embedding_dim, hidden_dim, quantum_strength, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "LSTMTagger"]
