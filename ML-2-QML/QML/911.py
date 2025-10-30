"""Quantum‑enhanced LSTM with attention and mixture weight.

The quantum part uses a small variational circuit per gate. The classical part remains identical to the ML module. A learnable mix_weight blends the two gate outputs. Multi‑head attention is applied after the sequence of hidden states.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

# --------------------------------------------------------------------------- #
# Quantum layer
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    """Small variational circuit producing n_qubits outputs."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode input into rotation angles
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "rz", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        # Trainable single‑qubit rotations
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, n_wires)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

# --------------------------------------------------------------------------- #
# Hybrid LSTM cell
# --------------------------------------------------------------------------- #
class QLSTMGen(nn.Module):
    """Hybrid LSTM cell with optional quantum gates, attention and mix weight."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        num_heads: int = 4,
        mix_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.num_heads = num_heads

        # Classical gates
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if n_qubits > 0:
            # Quantum gates
            self.forget_q = QLayer(n_qubits)
            self.input_q = QLayer(n_qubits)
            self.update_q = QLayer(n_qubits)
            self.output_q = QLayer(n_qubits)

            # Linear projections to quantum space
            self.forget_q_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.input_q_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.update_q_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.output_q_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

            # Mapping from quantum output to hidden_dim
            self.q_to_hidden = nn.Linear(n_qubits, hidden_dim)
        else:
            self.forget_q = None

        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Learnable mix weight
        self.mix_weight = nn.Parameter(torch.tensor(mix_weight))

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Classical gate outputs
            f_c = torch.sigmoid(self.forget_gate(combined))
            i_c = torch.sigmoid(self.input_gate(combined))
            g_c = torch.tanh(self.update_gate(combined))
            o_c = torch.sigmoid(self.output_gate(combined))

            if self.n_qubits > 0:
                # Quantum gate outputs
                f_q = self.forget_q(self.forget_q_lin(combined))
                i_q = self.input_q(self.input_q_lin(combined))
                g_q = self.update_q(self.update_q_lin(combined))
                o_q = self.output_q(self.output_q_lin(combined))

                f_q = self.q_to_hidden(f_q)
                i_q = self.q_to_hidden(i_q)
                g_q = self.q_to_hidden(g_q)
                o_q = self.q_to_hidden(o_q)

                # Blend
                f = self.mix_weight * f_c + (1 - self.mix_weight) * f_q
                i = self.mix_weight * i_c + (1 - self.mix_weight) * i_q
                g = self.mix_weight * g_c + (1 - self.mix_weight) * g_q
                o = self.mix_weight * o_c + (1 - self.mix_weight) * o_q
            else:
                f, i, g, o = f_c, i_c, g_c, o_c

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)

        # Apply multi‑head attention over the sequence
        attn_out, _ = self.attention(outputs, outputs, outputs)
        outputs = outputs + attn_out * self.mix_weight

        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
# Tagger wrapper
# --------------------------------------------------------------------------- #
class LSTMTaggerGen(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        num_heads: int = 4,
        mix_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            num_heads=num_heads,
            mix_weight=mix_weight,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen", "LSTMTaggerGen"]
