"""Quantum‑enhanced LSTM cell with parameter clipping and hybrid gate selection.

The module mirrors the classical `QLSTMGen` but implements the
gates as small variational circuits using `torchquantum`.  A
`use_quantum` flag allows the circuit to be disabled, falling back to
simple linear layers, which is useful for ablation studies or when a
full quantum backend is unavailable.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
#  Quantum gate wrapper
# --------------------------------------------------------------------------- #

class _QuantumGate(tq.QuantumModule):
    """Small variational circuit that takes an n‑qubit state vector and
    returns a measurement in the Pauli‑Z basis.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
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
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)


# --------------------------------------------------------------------------- #
#  Hybrid LSTM cell – quantum version
# --------------------------------------------------------------------------- #

class QLSTMGen(nn.Module):
    """Hybrid LSTM cell that can use classical linear gates or quantum circuits."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, use_quantum: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum

        gate_dim = hidden_dim
        if use_quantum:
            self.forget = _QuantumGate(n_wires=n_qubits)
            self.input = _QuantumGate(n_wires=n_qubits)
            self.update = _QuantumGate(n_wires=n_qubits)
            self.output = _QuantumGate(n_wires=n_qubits)

            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            self.forget = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.input = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.update = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.output = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.use_quantum:
                f = torch.sigmoid(self.forget(self.linear_forget(combined).clamp(-5.0, 5.0)))
                i = torch.sigmoid(self.input(self.linear_input(combined).clamp(-5.0, 5.0)))
                g = torch.tanh(self.update(self.linear_update(combined).clamp(-5.0, 5.0)))
                o = torch.sigmoid(self.output(self.linear_output(combined).clamp(-5.0, 5.0)))
            else:
                f = torch.sigmoid(self.forget(combined))
                i = torch.sigmoid(self.input(combined))
                g = torch.tanh(self.update(combined))
                o = torch.sigmoid(self.output(combined))
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
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
#  Sequence tagging model – quantum version
# --------------------------------------------------------------------------- #

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between the hybrid quantum LSTM and nn.LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits, use_quantum)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMGen", "LSTMTagger"]
