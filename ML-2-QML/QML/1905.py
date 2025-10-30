"""Quantum‑enhanced HybridLSTM using Pennylane for differentiable circuits."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QLayer(nn.Module):
    """
    Quantum layer that maps a classical vector to a quantum state,
    applies a parameterised circuit, and measures qubits in the Z basis.
    """

    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.params = nn.Parameter(torch.randn(n_qubits))
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor) -> torch.Tensor:
        for i, val in enumerate(x):
            qml.RX(val, wires=i)
            qml.RY(val, wires=i)
        for i in range(self.n_qubits):
            qml.RZ(self.params[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return torch.stack([qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return self.qnode(x)
        return torch.stack([self.qnode(xi) for xi in x])

class HybridLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM where each gate is computed by a quantum circuit.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.forget_q = QLayer(n_qubits)
        self.input_q = QLayer(n_qubits)
        self.update_q = QLayer(n_qubits)
        self.output_q = QLayer(n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f_q = self.forget_linear(combined)
            i_q = self.input_linear(combined)
            g_q = self.update_linear(combined)
            o_q = self.output_linear(combined)

            f = torch.sigmoid(self.forget_q(f_q))
            i = torch.sigmoid(self.input_q(i_q))
            g = torch.tanh(self.update_q(g_q))
            o = torch.sigmoid(self.output_q(o_q))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the quantum HybridLSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridLSTM", "LSTMTagger"]
