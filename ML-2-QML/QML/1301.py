import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np
from typing import Tuple

class HybridLSTMTagger(nn.Module):
    """
    Quantum‑enhanced LSTM‑based sequence tagger.
    Implements a variational quantum circuit for each LSTM gate.
    The circuit is parameter‑shift differentiable and runs on the
    Pennylane default simulator.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 8,
                 depth: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.n_qubits = n_qubits
        self.depth = depth

        # Classical linear layers that feed into the quantum circuit
        self.input_linear = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.forget_linear = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(embedding_dim + hidden_dim, n_qubits)

        # Quantum device and circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Parameterized gates
        self.forget_params = nn.Parameter(torch.randn(n_qubits, depth))
        self.input_params = nn.Parameter(torch.randn(n_qubits, depth))
        self.update_params = nn.Parameter(torch.randn(n_qubits, depth))
        self.output_params = nn.Parameter(torch.randn(n_qubits, depth))

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def _quantum_gate(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Apply a variational circuit to the input vector `x` and return
        the expectation values of PauliZ on each qubit.
        """
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, gate_params):
            # Encode classical data into rotation angles
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            # Variational layers
            for d in range(self.depth):
                for i in range(self.n_qubits):
                    qml.RZ(gate_params[i, d], wires=i)
                # Entangle
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Ensure inputs are of shape (n_qubits,)
        x = x.view(-1)
        if x.shape[0] < self.n_qubits:
            pad = torch.zeros(self.n_qubits - x.shape[0], device=x.device)
            x = torch.cat([x, pad])
        elif x.shape[0] > self.n_qubits:
            x = x[:self.n_qubits]
        return circuit(x, params)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of sentences.
        ``sentence`` is expected to be of shape (seq_len, batch).
        """
        embeds = self.word_embeddings(sentence)
        hx = torch.zeros(sentence.size(1), self.hidden_dim, device=embeds.device)
        cx = torch.zeros(sentence.size(1), self.hidden_dim, device=embeds.device)

        outputs = []
        for x in embeds:
            combined = torch.cat([x, hx], dim=1)
            # Classical linear projections to quantum input space
            f_in = self.forget_linear(combined)
            i_in = self.input_linear(combined)
            g_in = self.update_linear(combined)
            o_in = self.output_linear(combined)

            f = torch.sigmoid(self._quantum_gate(f_in, self.forget_params))
            i = torch.sigmoid(self._quantum_gate(i_in, self.input_params))
            g = torch.tanh(self._quantum_gate(g_in, self.update_params))
            o = torch.sigmoid(self._quantum_gate(o_in, self.output_params))

            # Classical cell state update
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        tag_logits = self.hidden2tag(outputs)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["HybridLSTMTagger"]
