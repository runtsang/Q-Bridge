import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QuantumLSTMCell(nn.Module):
    """
    Quantum‑based LSTM cell with a variational circuit for each gate.
    The quantum circuit uses Pennylane’s quantum device for simulation or real hardware.
    The cell is fully compatible with the classical LSTM cell interface.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        # Quantum device: one qubit per input feature
        self.device = qml.device("default.qubit", wires=n_qubits)
        # Classical linear layer to transform input to qubit space
        self.fc = nn.Linear(input_dim + hidden_dim, n_qubits)
        # Parameterised quantum circuits for each gate
        self.forget_gate = self._make_gate("forget")
        self.input_gate  = self._make_gate("input")
        self.update_gate = self._make_gate("update")
        self.output_gate = self._make_gate("output")
        # Linear layer to map quantum outputs to hidden dimension
        self.out_fc = nn.Linear(n_qubits, hidden_dim)
        # Optional dropout
        self.dropout = nn.Dropout(p=0.1)

    def _make_gate(self, name: str):
        """Create a variational circuit that returns a expectation value of Pauli‑Z for each qubit."""
        @qml.qnode(self.device, interface="torch")
        def circuit(x):
            # Encode input features via rotations
            for i in range(x.shape[1]):
                qml.RX(x[:, i], wires=i)
            # Simple entanglement pattern
            for i in range(x.shape[1] - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation values for each qubit
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(x.shape[1])]
        return circuit

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for a sequence of shape (seq_len, batch, input_dim)."""
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)          # (batch, input_dim+hidden)
            # Classical linear transform to qubit space
            q_in = self.fc(combined)                      # (batch, n_qubits)

            # Quantum gates
            f = torch.sigmoid(self.forget_gate(q_in))
            i = torch.sigmoid(self.input_gate(q_in))
            g = torch.tanh(self.update_gate(q_in))
            o = torch.sigmoid(self.output_gate(q_in))

            # Rescale to hidden dimension
            f = self.out_fc(f)
            i = self.out_fc(i)
            g = self.out_fc(g)
            o = self.out_fc(o)

            # LSTM recurrence
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.out_fc.out_features, device=device),
            torch.zeros(batch_size, self.out_fc.out_features, device=device),
        )

class QuantumLSTMTagger(nn.Module):
    """
    Sequence‑tagging model that uses the quantum LSTM cell.
    The API mirrors the original LSTMTagger: ``embedding_dim, hidden_dim,
    vocab_size, tagset_size`` and an optional ``n_qubits`` flag.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumLSTMCell(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)                # (seq_len, batch)
        # Convert to (seq_len, batch, embed_dim)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QuantumLSTMCell", "QuantumLSTMTagger"]
