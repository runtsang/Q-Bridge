import torch
import torch.nn as nn
import torch.nn.functional as F

class QLSTM(nn.Module):
    """
    Classical LSTM with residual gate connections and optional gate dropout.
    Each gate is computed as:
        gate = sigmoid(linear(x, h) + residual(h))
    where residual is a learned linear transform of the hidden state.
    Gate dropout can be applied during training to regularise the model.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, gate_dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.gate_dropout = gate_dropout

        # Gate linear layers
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Residual transforms
        self.forget_res = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.input_res = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_res = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_res = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(p=gate_dropout)

    def forward(self, inputs: torch.Tensor, states: tuple = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined) + self.forget_res(hx))
            i = torch.sigmoid(self.input_linear(combined) + self.input_res(hx))
            g = torch.tanh(self.update_linear(combined) + self.update_res(hx))
            o = torch.sigmoid(self.output_linear(combined) + self.output_res(hx))
            if self.training and self.gate_dropout > 0.0:
                f, i, g, o = [self.dropout(g) for g in (f, i, g, o)]
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the enhanced QLSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, gate_dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, gate_dropout=gate_dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
