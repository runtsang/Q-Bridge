"""Quantum‑LSTM with a variational ansatz and skip‑connection support.

The implementation mirrors the original quantum‑LSTM but replaces the
linear‑to‑quantum mapping with a trainable variational circuit (Ansatz)
that learns to map the classical state to a measurement output.
It also introduces a quantum‑skip‑connection that adds the quantum
output to the classical gate pre‑activation, providing richer
expressivity while keeping the same high‑level API.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QLayer(nn.Module):
    """Variational quantum layer that maps a classical vector to a quantum measurement."""
    def __init__(self, n_qubits: int, depth: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Trainable parameters for the ansatz
        self.params = nn.Parameter(torch.randn(depth, n_qubits, 3))

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor):
            # Encode inputs via RX rotations
            for i in range(n_qubits):
                qml.RX(inputs[i], wires=i)
            # Ansatz layers
            for d in range(depth):
                for i in range(n_qubits):
                    qml.RX(self.params[d, i, 0], wires=i)
                    qml.RY(self.params[d, i, 1], wires=i)
                    qml.RZ(self.params[d, i, 2], wires=i)
                # Entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            # Measure all qubits in Z basis
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_qubits)
        return self.circuit(x)  # returns (batch, n_qubits)

class QLSTM(nn.Module):
    """Quantum‑LSTM cell where gates are realised by variational circuits."""
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 depth: int = 1,
                 residual: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.residual = residual

        # Classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Linear mapping to quantum circuit input
        self.forget_q_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_q_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_q_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_q_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum layers
        self.forget_q = QLayer(n_qubits, depth=depth)
        self.input_q = QLayer(n_qubits, depth=depth)
        self.update_q = QLayer(n_qubits, depth=depth)
        self.output_q = QLayer(n_qubits, depth=depth)

        # Map quantum output to hidden dimension
        self.quantum_to_hidden = nn.Linear(n_qubits, hidden_dim)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Classical gate pre‑activations
            f_class = self.forget_linear(combined)
            i_class = self.input_linear(combined)
            g_class = self.update_linear(combined)
            o_class = self.output_linear(combined)

            # Quantum gate outputs
            f_q = self.forget_q(self.forget_q_linear(combined))
            i_q = self.input_q(self.input_q_linear(combined))
            g_q = self.update_q(self.update_q_linear(combined))
            o_q = self.output_q(self.output_q_linear(combined))

            # Map quantum outputs to hidden dimension and add to classical pre‑activations
            f = torch.sigmoid(f_class + self.quantum_to_hidden(f_q))
            i = torch.sigmoid(i_class + self.quantum_to_hidden(i_q))
            g = torch.tanh(g_class + self.quantum_to_hidden(g_q))
            o = torch.sigmoid(o_class + self.quantum_to_hidden(o_q))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        return torch.cat(outputs, dim=0), (hx, cx)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 depth: int = 1,
                 residual: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim,
                              hidden_dim,
                              n_qubits=n_qubits,
                              depth=depth,
                              residual=residual)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out = self.lstm(embeds.view(len(sentence), 1, -1))[0]
        return F.log_softmax(self.hidden2tag(lstm_out.view(len(sentence), -1)), dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
