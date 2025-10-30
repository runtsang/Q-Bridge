"""Quantum‑enhanced LSTM with real quantum circuits for gates and state updates."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class _QuantumGate(tq.QuantumModule):
    """Reusable quantum module for LSTM gates."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        # Entangle wires in a ring
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)


class QLSTM(nn.Module):
    """Hybrid classical‑quantum LSTM with optional quantum attention."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 8, mode: str = "quantum"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.mode = mode.lower()
        if self.mode not in {"classical", "quantum"}:
            raise ValueError("mode must be 'classical' or 'quantum'")

        # Linear projection to n_qubits * 4 for the four gates
        self.proj = nn.Linear(input_dim + hidden_dim, n_qubits * 4)

        # Quantum gate module reused for all gates
        self.qgate = _QuantumGate(n_qubits)
        self.qgate_proj = nn.Linear(n_qubits, hidden_dim)

        # Quantum attention module
        self.attn_qgate = _QuantumGate(n_qubits)
        self.attn_proj = nn.Linear(hidden_dim, n_qubits)
        self.attn_back = nn.Linear(n_qubits, hidden_dim)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
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
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            gate_inputs = self.proj(combined)  # (batch, n_qubits*4)
            f_in, i_in, g_in, o_in = gate_inputs.chunk(4, dim=1)

            if self.mode == "classical":
                f = torch.sigmoid(f_in)
                i = torch.sigmoid(i_in)
                g = torch.tanh(g_in)
                o = torch.sigmoid(o_in)
            else:
                f = torch.sigmoid(self.qgate_proj(self.qgate(f_in)))
                i = torch.sigmoid(self.qgate_proj(self.qgate(i_in)))
                g = torch.tanh(self.qgate_proj(self.qgate(g_in)))
                o = torch.sigmoid(self.qgate_proj(self.qgate(o_in)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            # Quantum‑controlled attention on the hidden state
            if self.mode == "quantum":
                hx_q = self.attn_qgate(self.attn_proj(hx))
                hx = self.attn_back(hx_q)

            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence tagging model with a quantum‑enhanced LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 8,
        mode: str = "quantum",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, mode=mode)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
