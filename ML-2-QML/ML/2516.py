"""Hybrid LSTM with optional quantum gates and feed‑forward angle generator."""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# TorchQuantum for quantum layers
import torchquantum as tq
import torchquantum.functional as tqf


class EstimatorQNN(nn.Module):
    """Feed‑forward network that generates rotation angles for quantum gates."""
    def __init__(self, input_dim: int, hidden_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QLayer(tq.QuantumModule):
    """Quantum gate implemented with a small variational circuit."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode each feature as an RX rotation
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        # Trainable RX gates
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for i, gate in enumerate(self.params):
            gate(qdev, wires=i)
        # Entangle neighbouring qubits
        for i in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[i, i + 1])
        return self.measure(qdev)


class QLSTM(nn.Module):
    """Hybrid LSTM cell with optional quantum gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim

        # Classical linear projections
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        if n_qubits > 0:
            # Quantum layers
            self.forget_q = QLayer(n_qubits)
            self.input_q = QLayer(n_qubits)
            self.update_q = QLayer(n_qubits)
            self.output_q = QLayer(n_qubits)
            # Map quantum output to hidden dimension
            self.quantum_to_hidden = nn.Linear(n_qubits, hidden_dim)
            # Feed‑forward network to generate angles
            self.angle_generator = EstimatorQNN(input_dim + hidden_dim, hidden_dim)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            # Classical gate activations
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            if self.n_qubits > 0:
                angles = self.angle_generator(combined)
                f = torch.sigmoid(self.quantum_to_hidden(self.forget_q(angles)))
                i = torch.sigmoid(self.quantum_to_hidden(self.input_q(angles)))
                g = torch.tanh(self.quantum_to_hidden(self.update_q(angles)))
                o = torch.sigmoid(self.quantum_to_hidden(self.output_q(angles)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence tagging model that supports classical or hybrid quantum LSTM."""
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
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
