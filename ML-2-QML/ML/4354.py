import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits=0,
                 use_transformer=False, transformer_params=None,
                 use_selfattention=False, use_sampler=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_transformer = use_transformer
        self.use_selfattention = use_selfattention
        self.use_sampler = use_sampler

        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if use_transformer:
            tp = transformer_params or {}
            self.transformer_block = TransformerBlockClassical(
                embed_dim=hidden_dim,
                num_heads=tp.get('num_heads', 4),
                ffn_dim=tp.get('ffn_dim', 4 * hidden_dim),
                dropout=tp.get('dropout', 0.1)
            )
        if use_selfattention:
            self.self_attention = ClassicalSelfAttention(embed_dim=hidden_dim)
        if use_sampler:
            self.sampler = SamplerModule()

    def forward(self, inputs, states=None):
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
        outputs = torch.cat(outputs, dim=0)
        if self.use_transformer:
            outputs = self.transformer_block(outputs)
        if self.use_selfattention:
            attn_out = self.self_attention.run(
                rotation_params=np.random.rand(self.hidden_dim, self.hidden_dim),
                entangle_params=np.random.rand(self.hidden_dim, self.hidden_dim),
                inputs=outputs.permute(1,0,2).numpy()
            )
            attn_out = torch.tensor(attn_out, device=outputs.device, dtype=outputs.dtype)
            outputs = outputs + attn_out.permute(1,0,2)
        if self.use_sampler:
            sampled = self.sampler(outputs[-1].unsqueeze(0))
            hx = sampled.squeeze(0)
        return outputs, (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,
                 n_qubits=0, use_transformer=False, transformer_params=None,
                 use_selfattention=False, use_sampler=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0 or use_transformer or use_selfattention or use_sampler:
            self.lstm = QLSTM(embedding_dim, hidden_dim,
                              n_qubits=n_qubits,
                              use_transformer=use_transformer,
                              transformer_params=transformer_params,
                              use_selfattention=use_selfattention,
                              use_sampler=use_sampler)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = q.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).transpose(1,2)
        k = k.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).transpose(1,2)
        v = v.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / np.sqrt(embed_dim // self.num_heads)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        attn = torch.matmul(scores, v)
        attn = attn.transpose(1,2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_linear(attn)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ClassicalSelfAttention:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim

    def run(self, rotation_params, entangle_params, inputs):
        query = torch.from_numpy(inputs @ rotation_params.reshape(self.embed_dim, -1)).float()
        key = torch.from_numpy(inputs @ entangle_params.reshape(self.embed_dim, -1)).float()
        value = torch.from_numpy(inputs).float()
        scores = F.softmax(torch.matmul(query, key.T) / np.sqrt(self.embed_dim), dim=-1)
        return (torch.matmul(scores, value)).numpy()

class SamplerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, inputs):
        return F.softmax(self.net(inputs), dim=-1)

__all__ = ["QLSTM", "LSTMTagger"]
