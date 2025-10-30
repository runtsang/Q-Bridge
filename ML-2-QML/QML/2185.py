"""Quantum-enhanced LSTM layers for sequence tagging."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuantumGate(nn.Module):
    """Variational quantum gate implemented with PennyLane."""
    def __init__(self, n_qubits: int, depth: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        # Parameters for each layer, qubit, and rotation
        self.params = nn.Parameter(torch.randn(self.depth, self.n_qubits, 3))
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        def circuit(inputs, params):
            for d in range(self.depth):
                for q in range(self.n_qubits):
                    qml.RX(inputs[q] + params[d, q, 0], wires=q)
                    qml.RY(inputs[q] + params[d, q, 1], wires=q)
                    qml.RZ(inputs[q] + params[d, q, 2], wires=q)
                for q in range(self.n_qubits - 1):
                    qml.CNOT(q, q + 1)
                qml.CNOT(self.n_qubits - 1, 0)
            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

        self.circuit = qml.QNode(circuit, self.dev, interface="torch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, hidden_dim)
        angles = torch.tanh(x)
        batch_size = angles.shape[0]
        out = []
        for i in range(batch_size):
            out.append(self.circuit(angles[i], self.params))
        out = torch.stack(out, dim=0)
        return out  # shape: (batch_size, n_qubits)

class QLSTM(nn.Module):
    """Quantum LSTM cell where gates are realized by variational quantum circuits."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        self.forget = QuantumGate(n_qubits, depth=self.depth)
        self.input = QuantumGate(n_qubits, depth=self.depth)
        self.update = QuantumGate(n_qubits, depth=self.depth)
        self.output = QuantumGate(n_qubits, depth=self.depth)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
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

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, depth=depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
