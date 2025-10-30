import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np
from typing import Tuple, Optional

class QLSTMQuantum(nn.Module):
    """Bidirectional, depth‑controlled quantum‑enhanced LSTM using PennyLane."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        bidirectional: bool = False,
        depth: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.bidirectional = bidirectional
        self.depth = depth

        self.q_device = qml.device("default.qubit", wires=n_qubits)
        self.q_circuit = self._build_circuit(depth)

        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if bidirectional:
            self.back_lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)

    def _build_circuit(self, depth: int):
        def circuit(x):
            for i in range(depth):
                qml.Hadamard(wires=range(self.n_qubits))
                qml.RX(x[0], wires=0)
                qml.RX(x[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(x[2], wires=2)
                qml.CNOT(wires=[1, 2])
                qml.RZ(x[3], wires=3)
            return qml.expval(qml.Z(0))
        return circuit

    def _quantum_gate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.q_circuit(x.cpu().numpy())], dtype=torch.float32)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self._quantum_gate(self.forget(combined))))
            i = torch.sigmoid(self.input(self._quantum_gate(self.input(combined))))
            g = torch.tanh(self.update(self._quantum_gate(self.update(combined))))
            o = torch.sigmoid(self.output(self._quantum_gate(self.output(combined))))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)

        if self.bidirectional:
            rev = torch.flip(outputs, dims=[0])
            rev_out, _ = self.back_lstm(rev.unsqueeze(1))
            rev_out = rev_out.squeeze(1)
            rev_out = torch.flip(rev_out, dims=[0])
            outputs = torch.cat([outputs, rev_out], dim=-1)

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
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTaggerQuantum(nn.Module):
    """Sequence tagging model with the quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        bidirectional: bool = False,
        depth: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = (
            QLSTMQuantum(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                bidirectional=bidirectional,
                depth=depth,
            )
            if n_qubits
            else nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        )
        self.hidden2tag = nn.Linear(hidden_dim * (2 if bidirectional else 1), tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMQuantum", "LSTMTaggerQuantum"]
