import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLayer(nn.Module):
    """Variational quantum circuit producing a single qubit measurement as gate value."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            # Encode inputs
            for i in range(self.n_qubits):
                qml.RX(inputs[:, i], wires=i)
            # Variational layer
            for i in range(self.n_qubits):
                qml.RY(weights[i], wires=i)
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit
        self.weights = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        return torch.sigmoid(self.circuit(x, self.weights))

class QLSTM(nn.Module):
    """Quantum‑augmented LSTM cell where each gate is realised by a variational circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        # Linear projections to qubit space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        # Quantum layers per gate
        self.forget_q = QLayer(n_qubits)
        self.input_q = QLayer(n_qubits)
        self.update_q = QLayer(n_qubits)
        self.output_q = QLayer(n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = self.forget_q(self.forget_lin(combined))
            i = self.input_q(self.input_lin(combined))
            g = torch.tanh(self.update_lin(combined))
            o = self.output_q(self.output_lin(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]
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
                 n_heads: int = 1,
                 use_residual: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=0)  # falls back to classical
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
