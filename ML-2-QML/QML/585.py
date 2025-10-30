import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLayer(nn.Module):
    """
    Parameterised quantum layer that encodes each feature via RX rotations,
    applies a trainable Ry rotation per qubit, entangles neighbouring qubits
    with CNOTs, and measures Pauli‑Z expectation values.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.params = nn.Parameter(torch.randn(n_qubits))
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev,
                               interface="torch", diff_method="backprop",
                               batch_processing=True)

    def _circuit(self, data: torch.Tensor, params: torch.Tensor):
        # data: (batch, n_qubits)
        for i in range(self.n_qubits):
            qml.RX(data[i], wires=i)
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.
        Returns a tensor of shape (batch, n_qubits).
        """
        return self.qnode(x, self.params)

class QLSTMGen(nn.Module):
    """
    Quantum‑enhanced LSTM cell. Gate values are computed by a small
    quantum circuit that operates on a low‑dimensional representation
    of the concatenated input and hidden state. The quantum output is
    then linearly mapped to the hidden dimension and combined with
    the classical LSTM recurrence.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, use_quantum: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum

        # Classical projections to a low‑dimensional quantum space
        self.forget_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_proj  = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_proj = nn.Linear(input_dim + hidden_dim, n_qubits)

        if self.use_quantum:
            # Quantum gates
            self.forget_qgate = QLayer(n_qubits)
            self.input_qgate  = QLayer(n_qubits)
            self.update_qgate = QLayer(n_qubits)
            self.output_qgate = QLayer(n_qubits)

            # Map quantum outputs back to hidden dimension
            self.forget_to_hidden = nn.Linear(n_qubits, hidden_dim)
            self.input_to_hidden  = nn.Linear(n_qubits, hidden_dim)
            self.update_to_hidden = nn.Linear(n_qubits, hidden_dim)
            self.output_to_hidden = nn.Linear(n_qubits, hidden_dim)
        else:
            # Fallback to classical linear gates
            self.forget_qgate = nn.Identity()
            self.input_qgate  = nn.Identity()
            self.update_qgate = nn.Identity()
            self.output_qgate = nn.Identity()
            self.forget_to_hidden = nn.Identity()
            self.input_to_hidden  = nn.Identity()
            self.update_to_hidden = nn.Identity()
            self.output_to_hidden = nn.Identity()

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            inputs: Tensor of shape (seq_len, batch, input_dim)
            states: Optional tuple (h, c) each of shape (batch, hidden_dim)

        Returns:
            outputs: Tensor of shape (seq_len, batch, hidden_dim)
            final states: (h, c)
        """
        h, c = self._init_states(inputs, states)
        outputs = []

        for x in torch.unbind(inputs, dim=0):
            combined = torch.cat([x, h], dim=1)

            # Classical projections to quantum space
            f_q = self.forget_proj(combined)
            i_q = self.input_proj(combined)
            g_q = self.update_proj(combined)
            o_q = self.output_proj(combined)

            # Quantum gate evaluation
            f_q = self.forget_qgate(f_q)
            i_q = self.input_qgate(i_q)
            g_q = self.update_qgate(g_q)
            o_q = self.output_qgate(o_q)

            # Map quantum output to hidden dimension
            f = torch.sigmoid(self.forget_to_hidden(f_q))
            i = torch.sigmoid(self.input_to_hidden(i_q))
            g = torch.tanh(self.update_to_hidden(g_q))
            o = torch.sigmoid(self.output_to_hidden(o_q))

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (h, c)

    def _init_states(self, inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTaggerGen(nn.Module):
    """
    Sequence tagging model that integrates :class:`QLSTMGen`.  When ``use_quantum``
    is False the model falls back to a standard ``nn.LSTM``.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, use_quantum: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if use_quantum and n_qubits > 0:
            self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits, use_quantum=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(logits, dim=1)

__all__ = ["QLSTMGen", "LSTMTaggerGen"]
