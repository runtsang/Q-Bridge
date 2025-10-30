"""Quantum‑enhanced LSTM cell that fuses classical linear gates with
parameter‑generating fully‑connected layers and quantum circuits.

The implementation follows the original QLSTM.py but enriches it with
a fully‑connected layer (FCL) that pre‑processes the linear output
before feeding it into the quantum circuit.  This mirrors the
classical side where the same FCL can be used to generate gate
parameters, thus providing a true hybrid scaling strategy.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

# --------------------------------------------------------------------------- #
# Helper: Fully connected layer
# --------------------------------------------------------------------------- #
class FCL(nn.Module):
    """Fully‑connected layer used to generate gate parameters for the quantum cell."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values)).mean(dim=0)

# --------------------------------------------------------------------------- #
# Quantum Layer
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    """Parameter‑ised quantum circuit used as a gate."""
    def __init__(self, n_wires: int) -> None:
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
# Quantum QLSTM
# --------------------------------------------------------------------------- #
class QLSTM__gen041(nn.Module):
    """Quantum LSTM cell that uses QLayer for each gate and an FCL to
    generate the parameters fed into the quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.fcl = FCL(n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f_raw = self.linear_forget(combined)
            i_raw = self.linear_input(combined)
            g_raw = self.linear_update(combined)
            o_raw = self.linear_output(combined)

            # Generate gate parameters via fully connected layer
            f_gate = self.fcl(f_raw)
            i_gate = self.fcl(i_raw)
            g_gate = self.fcl(g_raw)
            o_gate = self.fcl(o_raw)

            f = torch.sigmoid(self.forget(f_gate))
            i = torch.sigmoid(self.input(i_gate))
            g = torch.tanh(self.update(g_gate))
            o = torch.sigmoid(self.output(o_gate))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
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
# Quantum Tagger
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """Sequence‑tagging model that uses the hybrid quantum QLSTM__gen041."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM__gen041(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM__gen041", "LSTMTagger"]
