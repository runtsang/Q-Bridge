import pennylane as qml
import pennylane.numpy as pnp
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLinearGate(nn.Module):
    """Quantum‑parameterised linear gate using a variational circuit."""
    def __init__(self, in_features: int, out_features: int, n_wires: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_wires = n_wires
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.weight = nn.Parameter(torch.randn(n_wires))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map input to rotation angles
        def circuit(*params):
            for i, w in enumerate(self.weight):
                qml.RY(params[i], wires=i)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_wires)]

        qnode = qml.QNode(circuit, self.dev, interface="torch")
        qubit_out = qnode(x)
        # Linear mapping to desired output dimension
        return torch.matmul(qubit_out, torch.eye(self.out_features, device=x.device)) + self.bias


class QLSTMQuantum(nn.Module):
    """Quantum‑enhanced LSTM where each gate is a QLinearGate."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear projections to qubit space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin  = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates
        self.forget_gate = QLinearGate(n_qubits, hidden_dim, n_qubits)
        self.input_gate  = QLinearGate(n_qubits, hidden_dim, n_qubits)
        self.update_gate = QLinearGate(n_qubits, hidden_dim, n_qubits)
        self.output_gate = QLinearGate(n_qubits, hidden_dim, n_qubits)

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch, self.hidden_dim, device=device), torch.zeros(batch, self.hidden_dim, device=device)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)


class LSTMTaggerQuantum(nn.Module):
    """Sequence tagging model that uses the quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMQuantum(embedding_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLinearGate", "QLSTMQuantum", "LSTMTaggerQuantum"]
