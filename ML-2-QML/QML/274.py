"""Quantum‑enhanced LSTM layers for sequence tagging using Pennylane.

Each gate is realised by a variational circuit of configurable depth.
The input is encoded with RX rotations, followed by a series of Ry
rotations and CNOT entangling layers.  The circuit outputs expectation
values of Pauli‑Z, which are then passed through the gate activation
functions.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as np


class QLayer(nn.Module):
    """Variational quantum layer with a configurable depth."""

    def __init__(self, n_qubits: int, depth: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = max(1, depth)
        # Device for expectation‑value evaluation
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=0)

        # Parameter tensors for Ry rotations per layer
        self.rys = nn.Parameter(torch.randn(self.depth, self.n_qubits))
        # Parameter tensors for entanglement CNOT angles (not used, but kept for
        # future extensibility)
        self.cnot_params = nn.Parameter(torch.randn(self.depth, self.n_qubits - 1))

        # QNode that accepts a batch of inputs
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            # Encode each feature as an RX rotation
            for q in range(self.n_qubits):
                qml.RX(inputs[q], wires=q)

            # Variational layers
            for d in range(self.depth):
                for q in range(self.n_qubits):
                    qml.RY(self.rys[d, q], wires=q)
                # Entangling CNOT chain
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])

            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a batch of expectation values."""
        # x shape: (batch, n_qubits)
        return torch.stack([self.circuit(x[i]) for i in range(x.shape[0])], dim=0)


class QLSTM(nn.Module):
    """Drop‑in replacement for an LSTM that uses quantum gates."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        # Quantum layers for each gate
        self.forget = QLayer(n_qubits, depth)
        self.input_gate = QLayer(n_qubits, depth)
        self.update = QLayer(n_qubits, depth)
        self.output_gate = QLayer(n_qubits, depth)

        # Linear projections from (input + hidden) to qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Projection back to hidden dimension
        self.proj_back = nn.Linear(n_qubits, hidden_dim)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Linear projections
            f = self.linear_forget(combined)
            i = self.linear_input(combined)
            g = self.linear_update(combined)
            o = self.linear_output(combined)

            # Quantum gates
            f = torch.sigmoid(self.forget(f))
            i = torch.sigmoid(self.input_gate(i))
            g = torch.tanh(self.update(g))
            o = torch.sigmoid(self.output_gate(o))

            # Projection back to hidden dimension
            f = self.proj_back(f)
            i = self.proj_back(i)
            g = self.proj_back(g)
            o = self.proj_back(o)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        return torch.cat(outputs, dim=0), (hx, cx)


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
            self.lstm = QLSTM(
                embedding_dim, hidden_dim, n_qubits=n_qubits, depth=depth
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
