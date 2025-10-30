"""Quantum‑enhanced LSTM with PennyLane gates and configurable dropout and residuals."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple, Optional, List

class QLayer(nn.Module):
    """A small variational quantum circuit that maps an n‑qubit input to an n‑qubit output."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Parameters for the ansatz – one per qubit.
        self.params = nn.Parameter(torch.randn(n_qubits))
        # Quantum node with Torch interface for seamless autograd.
        self.qnode = qml.qnode(self._circuit, dev=self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Encode the classical input into a quantum state with RY gates.
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)
        # Apply a parameterised RY rotation on each qubit.
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        # Build a simple hardware‑efficient entangling layer.
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Measure all qubits in the Z basis.
        return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the expectation values of the Z operator for each qubit."""
        return self.qnode(x, self.params)


class QLSTM(nn.Module):
    """Quantum‑based LSTM cell where the four gates are realised by small
    variational quantum circuits.  Supports configurable dropout and
    optional residual connections.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dropout = dropout
        self.residual = residual

        # Quantum gates – one per layer.
        self.forget = nn.ModuleList([QLayer(n_qubits) for _ in range(n_layers)])
        self.input_gate = nn.ModuleList([QLayer(n_qubits) for _ in range(n_layers)])
        self.update = nn.ModuleList([QLayer(n_qubits) for _ in range(n_layers)])
        self.output = nn.ModuleList([QLayer(n_qubits) for _ in range(n_layers)])

        # Linear projections from (input + hidden) to the qubit dimension.
        self.linear_forget = nn.ModuleList(
            [nn.Linear(input_dim + hidden_dim, n_qubits) for _ in range(n_layers)]
        )
        self.linear_input = nn.ModuleList(
            [nn.Linear(input_dim + hidden_dim, n_qubits) for _ in range(n_layers)]
        )
        self.linear_update = nn.ModuleList(
            [nn.Linear(input_dim + hidden_dim, n_qubits) for _ in range(n_layers)]
        )
        self.linear_output = nn.ModuleList(
            [nn.Linear(input_dim + hidden_dim, n_qubits) for _ in range(n_layers)]
        )

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if states is not None:
            h, c = states
            return h, c
        batch_size = inputs.size(0)
        device = inputs.device
        h = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.n_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.n_layers)]
        return h, c

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Run the quantum‑enhanced LSTM over a sequence.

        Args:
            inputs: Tensor of shape (seq_len, batch, input_dim).
            states: Optional tuple of hidden and cell states for each layer.

        Returns:
            output: Tensor of shape (seq_len, batch, hidden_dim) – the output
                of the last layer.
            (h, c): Final hidden and cell states for each layer.
        """
        h, c = self._init_states(inputs, states)
        outputs = []

        for t in range(inputs.size(0)):
            x = inputs[t]
            new_h, new_c = [], []
            for i in range(self.n_layers):
                combined = torch.cat([x, h[i]], dim=1)
                f = torch.sigmoid(self.forget[i](self.linear_forget[i](combined)))
                i_g = torch.sigmoid(self.input_gate[i](self.linear_input[i](combined)))
                g = torch.tanh(self.update[i](self.linear_update[i](combined)))
                o = torch.sigmoid(self.output[i](self.linear_output[i](combined)))

                c_i = f * c[i] + i_g * g
                h_i = o * torch.tanh(c_i)

                if self.dropout > 0.0:
                    h_i = F.dropout(h_i, p=self.dropout, training=self.training)
                if self.residual:
                    h_i = h_i + x  # residual on the input of the layer

                new_h.append(h_i)
                new_c.append(c_i)
                x = h_i
            h, c = new_h, new_c
            outputs.append(h[-1].unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (h, c)


class LSTMTagger(nn.Module):
    """Sequence tagging model that can use either the quantum LSTM
    defined above or a standard classical ``nn.LSTM``."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        n_layers: int = 1,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                n_layers=n_layers,
                dropout=dropout,
                residual=residual,
            )
        else:
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=False,
            )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)


__all__ = ["QLSTM", "LSTMTagger"]
