"""
HybridQLSTM – quantum‑enhanced LSTM core with a quantum expectation head.

The quantum implementation mirrors the classical API but replaces
the linear gates with small quantum circuits (torchquantum).
An optional quantum head can be attached to the final classification
layer, providing a differentiable expectation value that can be
executed on real hardware or a simulator.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
#  Quantum gate layer used inside the quantum LSTM
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    """Parameterised quantum layer: RX gates followed by a CNOT chain."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)


# --------------------------------------------------------------------------- #
#  Quantum LSTM cell
# --------------------------------------------------------------------------- #
class QuantumQLSTM(nn.Module):
    """LSTM cell whose gates are realised by QLayer circuits."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim, self.hidden_dim, self.n_qubits = input_dim, hidden_dim, n_qubits
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _step(self, inputs: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([inputs, hx], dim=1)
        f = torch.sigmoid(self.forget(self.linear_forget(combined)))
        i = torch.sigmoid(self.input(self.linear_input(combined)))
        g = torch.tanh(self.update(self.linear_update(combined)))
        o = torch.sigmoid(self.output(self.linear_output(combined)))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = inputs.size()
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        outputs = []
        for t in range(seq_len):
            hx, cx = self._step(inputs[:, t, :], hx, cx)
            outputs.append(hx.unsqueeze(1))
        return torch.cat(outputs, dim=1), (hx, cx)


# --------------------------------------------------------------------------- #
#  Quantum expectation head
# --------------------------------------------------------------------------- #
class QuantumHybridLayer(nn.Module):
    """
    Maps a linear output to a probability via a single‑qubit rotation
    followed by measurement of Pauli‑Z.  The expectation value is passed
    through a sigmoid to obtain a probability in (0,1).
    """
    def __init__(self, in_features: int, n_qubits: int = 1, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, n_qubits)
        self.quantum_layer = QLayer(n_qubits)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.linear(x) + self.shift
        out = self.quantum_layer(angles)
        return torch.sigmoid(out.squeeze(-1))


# --------------------------------------------------------------------------- #
#  HybridQLSTM that switches between classical and quantum paths
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    Hybrid sequence tagger that swaps between classical and quantum gates
    and optionally replaces the final dense head with a quantum expectation
    layer.  The API matches the classical implementation so the same
    training loop can be used.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum_head: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.use_quantum_head = use_quantum_head
        if use_quantum_head:
            self.quantum_head = QuantumHybridLayer(tagset_size)
        else:
            self.quantum_head = None

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: LongTensor of shape (batch, seq_len)
        Returns:
            Either log‑softmax scores or a two‑class probability vector
            depending on `use_quantum_head`.
        """
        embeds = self.embedding(sentence)  # (batch, seq_len, embedding_dim)
        outputs, _ = self.lstm(embeds)     # (batch, seq_len, hidden_dim)
        tag_logits = self.hidden2tag(outputs)  # (batch, seq_len, tagset_size)
        if self.use_quantum_head:
            probs = self.quantum_head(tag_logits)  # (batch, seq_len)
            return torch.stack([probs, 1 - probs], dim=-1)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["HybridQLSTM"]
