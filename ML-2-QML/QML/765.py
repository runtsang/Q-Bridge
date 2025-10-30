import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class QGate(nn.Module):
    """Variational quantum gate with optional depolarizing noise."""
    def __init__(self, n_qubits: int, noise_prob: float = 0.0) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.noise_prob = noise_prob
        self.params = nn.Parameter(torch.randn(n_qubits, 3))
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        @qml.qnode(self.dev, interface="torch")
        def circuit(x_batch):
            for i in range(self.n_qubits):
                qml.RX(x_batch[:, i], wires=i)
                qml.RY(self.params[i, 0], wires=i)
                qml.RZ(self.params[i, 1], wires=i)
                if self.noise_prob > 0.0:
                    qml.DepolarizingChannel(self.noise_prob, wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                if self.noise_prob > 0.0:
                    qml.DepolarizingChannel(self.noise_prob, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit(x)

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell where gates are realized by small variational circuits."""
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 dropout: float = 0.0,
                 noise_prob: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)
        self.noise_prob = noise_prob

        self.forget_gate = QGate(n_qubits, noise_prob=noise_prob)
        self.input_gate = QGate(n_qubits, noise_prob=noise_prob)
        self.update_gate = QGate(n_qubits, noise_prob=noise_prob)
        self.output_gate = QGate(n_qubits, noise_prob=noise_prob)

        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

            hx = self.dropout(hx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can run both quantum and classical baselines side‑by‑side."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 dropout: float = 0.0,
                 noise_prob: float = 0.0) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim,
                          hidden_dim,
                          n_qubits=hidden_dim,
                          dropout=dropout,
                          noise_prob=noise_prob)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self,
                sentence: torch.Tensor,
                return_both: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        quantum_out = F.log_softmax(tag_logits, dim=1)

        if return_both:
            # Classical baseline
            classical_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
            lstm_out_cls, _ = classical_lstm(embeds.unsqueeze(0))
            tag_logits_cls = self.hidden2tag(lstm_out_cls.squeeze(0))
            classical_out = F.log_softmax(tag_logits_cls, dim=1)
            return quantum_out, classical_out

        return quantum_out

__all__ = ["QLSTM", "LSTMTagger"]
