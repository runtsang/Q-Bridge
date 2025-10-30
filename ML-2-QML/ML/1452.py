import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLinearGate(nn.Module):
    """
    Lightweight linear gate that can act classically or use a quantum placeholder.
    The quantum branch is currently a simple linear layer; replace with a
    variational circuit when integrating a quantum backend.
    """
    def __init__(self, in_features: int, out_features: int, use_quantum: bool = False):
        super().__init__()
        self.use_quantum = use_quantum
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class QLSTMCell(nn.Module):
    """
    Single LSTM cell where each gate is a QLinearGate.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 4, use_quantum: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum
        self.dropout = nn.Dropout(dropout)

        self.forget_gate = QLinearGate(input_dim + hidden_dim, hidden_dim, use_quantum)
        self.input_gate = QLinearGate(input_dim + hidden_dim, hidden_dim, use_quantum)
        self.update_gate = QLinearGate(input_dim + hidden_dim, hidden_dim, use_quantum)
        self.output_gate = QLinearGate(input_dim + hidden_dim, hidden_dim, use_quantum)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((x, h_prev), dim=1)
        f = torch.sigmoid(self.forget_gate(combined))
        i = torch.sigmoid(self.input_gate(combined))
        g = torch.tanh(self.update_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        h = self.dropout(h)
        return h, c

class QLSTM(nn.Module):
    """
    Multi‑layer LSTM stack with optional quantum gates.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 n_qubits: int = 4, use_quantum: bool = False, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            in_dim = input_dim if l == 0 else hidden_dim
            self.layers.append(
                QLSTMCell(in_dim, hidden_dim, n_qubits=n_qubits,
                          use_quantum=use_quantum, dropout=dropout)
            )

    def forward(self, inputs: torch.Tensor,
                h_0: Optional[torch.Tensor] = None,
                c_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_len, batch_size, _ = inputs.shape
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=inputs.device)
        if c_0 is None:
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=inputs.device)

        outputs = []
        h_n = []
        c_n = []

        for t in range(seq_len):
            x = inputs[t]
            h_t = []
            c_t = []
            for l, layer in enumerate(self.layers):
                h, c = layer(x, h_0[l], c_0[l])
                x = h
                h_t.append(h)
                c_t.append(c)
            outputs.append(x.unsqueeze(0))
            h_n.append(torch.stack(h_t))
            c_n.append(torch.stack(c_t))

        outputs = torch.cat(outputs, dim=0)
        h_n = torch.stack(h_n, dim=0)
        c_n = torch.stack(c_n, dim=0)
        return outputs, (h_n, c_n)

class QLSTMGen(nn.Module):
    """
    Sequence tagging model that can switch between classical and quantum LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 num_layers: int = 2, n_qubits: int = 4, use_quantum: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                          n_qubits=n_qubits, use_quantum=use_quantum, dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        sentence: (seq_len, batch)
        """
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTMGen", "QLSTM", "QLSTMCell", "QLinearGate"]
