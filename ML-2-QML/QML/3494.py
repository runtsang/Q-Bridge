"""Quantum‑enhanced LSTM module with optional sampler.

The implementation follows the structure of the classical
``QLSTMGen113`` but replaces the linear gates with
parameterised quantum circuits.  A small quantum sampler
module is also provided to illustrate how a quantum
pre‑processing step can be coupled to the LSTM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QSampler(tq.QuantumModule):
    """Parameterised quantum sampler that maps a 2‑dimensional input
    to a 2‑dimensional probability distribution.

    The circuit consists of a single rotation on each qubit
    followed by a trainable RX gate and a measurement in the
    Pauli‑Z basis.  The resulting measurement outcomes are
    converted to probabilities using ``softmax``.
    """
    def __init__(self, n_wires: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
            ]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution for each batch element."""
        qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                bsz=x.shape[0],
                                device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        meas = self.measure(qdev)  # shape (batch, n_wires)
        # Convert raw measurement (-1,1) to probabilities
        probs = F.softmax(meas, dim=-1)
        return probs

class QLayer(tq.QuantumModule):
    """Small quantum circuit used as a gate in the LSTM."""
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
        qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                bsz=x.shape[0],
                                device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

class QLSTMGen113(nn.Module):
    """Quantum‑enhanced LSTM.  When ``n_qubits`` is zero the class
    falls back to a classical implementation identical to
    the one in the ``ml_code`` module.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 0, use_sampler: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_sampler = use_sampler

        if self.n_qubits > 0:
            # Quantum gates
            self.forget = QLayer(self.n_qubits)
            self.input = QLayer(self.n_qubits)
            self.update = QLayer(self.n_qubits)
            self.output = QLayer(self.n_qubits)

            self.linear_forget = nn.Linear(input_dim + hidden_dim, self.n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, self.n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, self.n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, self.n_qubits)
        else:
            # Classical fallback
            gate_dim = hidden_dim
            self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        if self.use_sampler:
            self.sampler = QSampler()
        else:
            self.sampler = None

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            # Optional quantum sampler pre‑processing
            if self.sampler is not None:
                # Map input to 2‑dim vector for the sampler
                proj = x[:, :2] if x.shape[1] >= 2 else F.pad(x, (0, 2 - x.shape[1]))
                samp = self.sampler(proj)  # shape (batch, 2)
                # Broadcast and modulate the input
                samp_exp = samp.unsqueeze(-1).expand(-1, self.hidden_dim)
                x = x * samp_exp

            combined = torch.cat([x, hx], dim=1)

            if self.n_qubits > 0:
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
            else:
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the quantum LSTM when
    ``n_qubits`` is non‑zero or the sampler is enabled.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0 or use_sampler:
            self.lstm = QLSTMGen113(embedding_dim, hidden_dim,
                                    n_qubits=n_qubits,
                                    use_sampler=use_sampler)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen113", "LSTMTagger"]
