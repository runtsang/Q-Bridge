"""Quantum‑enhanced LSTM with depth‑controlled variational gates.

The class name `QLSTMGen` matches the classical implementation above.
It replaces the hard‑coded unitary per gate with a PennyLane variational
circuit of adjustable depth.  A hybrid loss can be computed by
combining the classical loss with a quantum expectation value.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple, Optional

class _QLayer(nn.Module):
    def __init__(self, n_qubits: int, depth: int, device: str = "cpu") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.params = nn.Parameter(torch.randn(depth, n_qubits))
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=None)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, inputs: torch.Tensor, params: torch.Tensor):
        for i in range(self.n_qubits):
            qml.RX(inputs[:, i], wires=i)
        for d in range(self.depth):
            for i in range(self.n_qubits):
                qml.RX(params[d, i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        raw = self.qnode(inputs, self.params)
        out = torch.stack(raw, dim=-1)
        if out.ndim == 2 and out.shape[0] == 1:
            return out.squeeze(0)
        return out

class QLSTMGen(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
        depth: int = 1,
        use_hybrid_loss: bool = False,
        loss_weight: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if hidden_dim!= n_qubits:
            raise ValueError("hidden_dim must equal n_qubits for the quantum gate outputs.")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.n_qubits = n_qubits
        self.depth = depth
        self.use_hybrid_loss = use_hybrid_loss
        self.loss_weight = loss_weight
        self.device = device

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear_forget = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(embedding_dim + hidden_dim, n_qubits)

        self.forget_q = _QLayer(n_qubits, depth, device)
        self.input_q = _QLayer(n_qubits, depth, device)
        self.update_q = _QLayer(n_qubits, depth, device)
        self.output_q = _QLayer(n_qubits, depth, device)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)  # (seq_len, embedding_dim)
        hx = torch.zeros(self.hidden_dim, device=embeds.device)
        cx = torch.zeros(self.hidden_dim, device=embeds.device)
        outputs = []
        for x in embeds.unbind(dim=0):
            combined = torch.cat([x, hx], dim=0)  # (embedding_dim + hidden_dim,)
            f = torch.sigmoid(self.forget_q(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_q(self.linear_input(combined)))
            g = torch.tanh(self.update_q(self.linear_update(combined)))
            o = torch.sigmoid(self.output_q(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        lstm_out = torch.cat(outputs, dim=0)  # (seq_len, hidden_dim)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)

    def get_hybrid_loss(self, classical_loss: torch.Tensor, quantum_reg: torch.Tensor) -> torch.Tensor:
        if not self.use_hybrid_loss:
            return classical_loss
        return classical_loss + self.loss_weight * quantum_reg

__all__ = ["QLSTMGen"]
