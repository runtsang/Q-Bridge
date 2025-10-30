"""Quantum‑enhanced LSTM with hybrid variational gates and Pennylane simulator support."""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class VariationalGate(nn.Module):
    """A small variational circuit that maps a classical vector to a qubit expectation vector."""
    def __init__(self, n_wires: int, depth: int = 1):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=n_wires, shots=0)

        # Parameters for each rotation in the circuit
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(1, requires_grad=True)) for _ in range(n_wires * depth)]
        )

        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            # initialise all qubits in |0> via Hadamard to create superposition
            for w in range(n_wires):
                qml.Hadamard(w)
            # apply rotations and entanglement
            for idx, param in enumerate(self.params):
                w = idx % n_wires
                qml.RZ(x[w], wires=w)
                qml.RZ(param, wires=w)
            for w in range(n_wires - 1):
                qml.CNOT(wires=[w, w + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(n_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a batch of inputs.

        Args:
            x: Tensor of shape (batch, n_wires) containing classical angles.

        Returns:
            Tensor of shape (batch, n_wires) with expectation values.
        """
        batch = x.shape[0]
        outputs = []
        for i in range(batch):
            outputs.append(self.circuit(x[i]))
        return torch.stack(outputs)


class QLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int,
                 depth: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        # For quantum case, hidden_dim must match n_qubits
        self.hidden_dim = hidden_dim if n_qubits == 0 else n_qubits
        self.depth = depth

        if n_qubits > 0:
            # Quantum gates
            self.forget_gate = VariationalGate(n_qubits, depth)
            self.input_gate = VariationalGate(n_qubits, depth)
            self.update_gate = VariationalGate(n_qubits, depth)
            self.output_gate = VariationalGate(n_qubits, depth)

            # Linear projections to qubit space
            self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            # Classical LSTM fallback
            self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            if self.n_qubits > 0:
                f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
                i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
                g = torch.tanh(self.update_gate(self.update_linear(combined)))
                o = torch.sigmoid(self.output_gate(self.output_linear(combined)))
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
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 depth: int = 1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim,
                              hidden_dim,
                              n_qubits=n_qubits,
                              depth=depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
