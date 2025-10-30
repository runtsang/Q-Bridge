import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QuantumGate(nn.Module):
    """Simulated quantum gate using a small MLP."""
    def __init__(self, input_dim: int, output_dim: int, depth: int = 1):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(input_dim, input_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QLSTM(nn.Module):
    """Hybrid LSTM with linear gates blended with a simulated quantum gate."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, depth: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.dropout = nn.Dropout(dropout)

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        # Simulated quantum gates
        self.forget_qgate = QuantumGate(gate_dim, gate_dim, depth)
        self.input_qgate = QuantumGate(gate_dim, gate_dim, depth)
        self.update_qgate = QuantumGate(gate_dim, gate_dim, depth)
        self.output_qgate = QuantumGate(gate_dim, gate_dim, depth)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined) + self.forget_qgate(combined))
            i = torch.sigmoid(self.input_linear(combined) + self.input_qgate(combined))
            g = torch.tanh(self.update_linear(combined) + self.update_qgate(combined))
            o = torch.sigmoid(self.output_linear(combined) + self.output_qgate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.dropout(hx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTagger(nn.Module):
    """Sequence tagging using hybrid or classical LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0, depth: int = 1, dropout: float = 0.0):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, depth=depth, dropout=dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTM", "LSTMTagger"]
