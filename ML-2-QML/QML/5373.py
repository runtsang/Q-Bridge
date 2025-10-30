"""Quantum‑enhanced LSTM with quantum gates and a quantum‑augmented attention decoder.

The implementation mirrors the classical `HybridQLSTMTagger` but replaces
classical linear gates with variational quantum circuits and the decoder
with a quantum‑augmented self‑attention block.  All interfaces remain
identical so the class can be used as a drop‑in replacement when a
quantum backend is available.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QGate(tq.QuantumModule):
    """Variational quantum circuit producing a vector of length `n_qubits`."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_qubits)
        returns: (batch, n_qubits)
        """
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for gate in self.params:
            gate(qdev)
        return self.measure(qdev)


class QuantumQLSTM(tq.QuantumModule):
    """Hybrid LSTM cell with quantum gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates
        self.forget_gate = QGate(n_qubits)
        self.input_gate = QGate(n_qubits)
        self.update_gate = QGate(n_qubits)
        self.output_gate = QGate(n_qubits)

        # Linear layers to map input+hidden to n_qubits
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        inputs: (seq_len, batch, input_dim)
        returns: (seq_len, batch, hidden_dim), (hx, cx)
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        lstm_out = torch.cat(outputs, dim=0)
        return lstm_out, (hx, cx)


class QuantumAttention(tq.QuantumModule):
    """Self‑attention block with quantum augmentation."""
    def __init__(self, embed_dim: int, n_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_k = embed_dim // n_heads

        # Classical projections
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Quantum module for attention output
        self.q_gate = QGate(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (seq_len, batch, embed_dim)
        returns: (seq_len, batch, embed_dim)
        """
        batch_size = x.size(1)
        seq_len = x.size(0)
        q = self.linear_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim).transpose(0, 1)

        # Quantum augmentation
        out = []
        for token in attn_output.unbind(dim=0):
            token = token.squeeze(0)  # (batch, embed_dim)
            mod = self.q_gate(token)
            out.append(mod.unsqueeze(0))
        attn_output = torch.cat(out, dim=0)
        return attn_output


class HybridQLSTM(tq.QuantumModule):
    """Placeholder for compatibility; not used in quantum implementation."""
    pass


class HybridQLSTMTagger(tq.QuantumModule):
    """Quantum‑enhanced sequence tagging model."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits)
        self.decoder = QuantumAttention(hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        sentence: (seq_len, batch)
        returns log‑softmaxed tag logits
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        decoded = self.decoder(lstm_out)
        logits = self.hidden2tag(decoded)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQLSTM", "HybridQLSTMTagger"]
