import torch
import torch.nn as nn
import torch.nn.functional as F

class QLSTM(nn.Module):
    """Classical LSTM cell with optional weight clipping and dropout."""
    def __init__(self, input_dim, hidden_dim, clip_weights=False, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.clip_weights = clip_weights
        self.dropout = dropout
        # Single linear transforms all gates at once
        self.linear = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        if clip_weights:
            self._clip_weights()

    def _clip_weights(self):
        with torch.no_grad():
            self.linear.weight.data.clamp_(-5.0, 5.0)
            self.linear.bias.data.clamp_(-5.0, 5.0)

    def forward(self, inputs, states=None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            gates = self.linear(combined)
            f, i, g, o = gates.chunk(4, dim=1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            if self.dropout > 0.0:
                hx = F.dropout(hx, self.dropout, training=self.training)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that can use the improved QLSTM."""
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,
                 clip_weights=False, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, clip_weights=clip_weights,
                          dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
