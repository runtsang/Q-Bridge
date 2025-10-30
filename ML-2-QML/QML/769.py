import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class QLSTMPlus(nn.Module):
    """
    Quantum LSTM cell where each gate is a variational quantum circuit.
    The interface matches QLSTMPlus in the classical module.
    """
    class _QGate(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, gate_name: str) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_qubits = n_qubits
            self.gate_name = gate_name

            # Map classical input to qubit rotation angles
            self.input_mapper = nn.Linear(input_dim, n_qubits)
            self.dev = qml.device("default.qubit", wires=n_qubits)
            self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="backprop")
            self.temperature = Parameter(torch.tensor(1.0))

        def _circuit(self, x: torch.Tensor) -> list[torch.Tensor]:
            # x shape: (batch, n_qubits)
            for i in range(self.n_qubits):
                qml.RY(x[:, i], wires=i)
            # Simple entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mapped = self.input_mapper(x)
            out = self.qnode(mapped)  # shape: (batch, n_qubits)
            out = out / self.temperature
            if self.gate_name == "update":
                return torch.tanh(out)
            else:
                return torch.sigmoid(out)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)

        # Quantum gates for each LSTM component
        self.forget_gate = self._QGate(input_dim + hidden_dim, hidden_dim, n_qubits, "forget")
        self.input_gate = self._QGate(input_dim + hidden_dim, hidden_dim, n_qubits, "input")
        self.update_gate = self._QGate(input_dim + hidden_dim, hidden_dim, n_qubits, "update")
        self.output_gate = self._QGate(input_dim + hidden_dim, hidden_dim, n_qubits, "output")

    def _init_states(self, inputs: torch.Tensor, states: tuple | None = None) -> tuple:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = self.forget_gate(combined)
            i = self.input_gate(combined)
            g = self.update_gate(combined)
            o = self.output_gate(combined)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.dropout(hx)
            outputs.append(hx.unsqueeze(0))

        return torch.cat(outputs, dim=0), (hx, cx)


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the quantum QLSTMPlus cell.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMPlus(embedding_dim, hidden_dim, n_qubits, dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMPlus", "LSTMTagger"]
