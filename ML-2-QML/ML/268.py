import torch
import torch.nn as nn
import torch.nn.functional as F

class QLSTM(nn.Module):
    """Hybrid classical LSTM with multi‑head attention and optional quantum gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, attn_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Attention module
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attn_heads, batch_first=True)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        seq_len = inputs.shape[0]
        for t in range(seq_len):
            x = inputs[t]
            combined = torch.cat([x, hx], dim=-1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        hidden_states = torch.cat(outputs, dim=0)  # [seq_len, batch, hidden_dim]
        # Apply attention across the sequence
        attn_output, _ = self.attn(hidden_states, hidden_states, hidden_states)
        # Fuse attention output with hidden states
        fused = hidden_states + attn_output
        return fused, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model supporting classical or hybrid LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, attn_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, attn_heads=attn_heads)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds.unsqueeze(1))
            lstm_out = lstm_out.squeeze(1)
        else:
            lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
