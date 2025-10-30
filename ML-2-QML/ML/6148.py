import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalSelfAttention(nn.Module):
    """
    Classical multi‑head self‑attention block.
    Uses linear projections for query, key, value and a soft‑max
    to compute attention weights.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, embed_dim)
        returns: (batch, seq_len, embed_dim)
        """
        batch, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)

class ClassicalQLSTM(nn.Module):
    """
    Classical LSTM cell that can optionally replace each gate
    with a quantum‑inspired linear transformation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.n_qubits = n_qubits

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
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

class HybridQLSTMAttention(nn.Module):
    """
    End‑to‑end model that first applies a classical self‑attention block
    and then processes the attended representations with a classical LSTM.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int,
                 num_classes: int, seq_len: int, n_qubits: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = ClassicalSelfAttention(embed_dim)
        self.lstm = ClassicalQLSTM(embed_dim, hidden_dim, n_qubits=n_qubits)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)
        returns logits: (batch, num_classes)
        """
        embeds = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        attn_out = self.attention(embeds)   # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(attn_out.permute(1, 0, 2))  # (seq_len, batch, hidden_dim)
        logits = self.classifier(lstm_out.mean(dim=0))
        return logits
