import torch
import torch.nn as nn
import torch.nn.functional as F

class QLSTM(nn.Module):
    """Classical LSTM with optional dropout on recurrent connections and
    gate‑level dropout. Designed as a drop‑in replacement for the
    original QLSTM while providing richer regularisation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0,
                 dropout: float = 0.0, gate_dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = dropout
        self.gate_dropout = gate_dropout

        # Classical linear gates
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        # Optional recurrent dropout
        self.rnn_dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            if self.rnn_dropout is not None:
                hx = self.rnn_dropout(hx)
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            if self.gate_dropout > 0.0:
                f = F.dropout(f, p=self.gate_dropout, training=self.training)
                i = F.dropout(i, p=self.gate_dropout, training=self.training)
                g = F.dropout(g, p=self.gate_dropout, training=self.training)
                o = F.dropout(o, p=self.gate_dropout, training=self.training)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """Sequence tagging with a classical LSTM that can be configured
    to use dropout, recurrent dropout or a quantum block via the
    ``n_qubits`` flag."""
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, dropout: float = 0.0,
                 gate_dropout: float = 0.0):
        super().__init__()
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim,
                              n_qubits=n_qubits,
                              dropout=dropout,
                              gate_dropout=gate_dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.unsqueeze(0).transpose(0, 1))
        return F.log_softmax(self.hidden2tag(lstm_out.squeeze(0)), dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
