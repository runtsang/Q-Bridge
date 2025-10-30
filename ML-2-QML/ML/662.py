import torch
import torch.nn as nn
import torch.nn.functional as F

class QLSTM(nn.Module):
    """
    Classical LSTM cell with dropout regularisation and a residual connection.
    Each gate is computed from a linear projection followed by dropout.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)

        # Linear projections for each gate
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Residual connection
        self.residual_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None) -> tuple:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f_in = self.forget_linear(combined)
            i_in = self.input_linear(combined)
            g_in = self.update_linear(combined)
            o_in = self.output_linear(combined)

            f = torch.sigmoid(f_in + self.dropout(f_in))
            i = torch.sigmoid(i_in + self.dropout(i_in))
            g = torch.tanh(g_in + self.dropout(g_in))
            o = torch.sigmoid(o_in + self.dropout(o_in))

            cx = f * cx + i * g
            hx_res = self.residual_linear(hx)
            hx = o * torch.tanh(cx) + hx_res * 0.1
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None = None) -> tuple:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between the enhanced classical LSTM
    and the standard nn.LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits, dropout=dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
