import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
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

        if n_qubits > 0:
            self.forget = QLayer(n_qubits)
            self.input = QLayer(n_qubits)
            self.update = QLayer(n_qubits)
            self.output = QLayer(n_qubits)
            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if use_transformer:
            tp = transformer_params or {}
            self.transformer_block = TransformerBlockQuantum(
                embed_dim=hidden_dim,
                num_heads=tp.get('num_heads', 4),
                ffn_dim=tp.get('ffn_dim', 4 * hidden_dim),
                n_qubits_transformer=tp.get('n_qubits_transformer', n_qubits),
                n_qubits_ffn=tp.get('n_qubits_ffn', n_qubits),
                n_qlayers=tp.get('n_qlayers', 1),
                q_device=tp.get('q_device', None),
                dropout=tp.get('dropout', 0.1)
            )
        if use_selfattention:
            self.self_attention = QuantumSelfAttention(n_qubits=4)
        if use_sampler:
            self.sampler = QuantumSamplerQNN()

    def forward(self, inputs, states=None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.n_qubits > 0:
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
            else:
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
                backend=None,
                rotation_params=np.random.rand(4*4).astype(np.float64),
                entangle_params=np.random.rand(4).astype(np.float64),
                inputs=outputs.permute(1,0,2).numpy(),
                shots=1024
            )
            attn_out = torch.tensor(attn_out, device=outputs.device, dtype=outputs.dtype)
            outputs = outputs + attn_out.permute(1,0,2)
        if self.use_sampler:
            sampled = self.sampler.run(
                backend=None,
                inputs=outputs[-1].cpu().numpy(),
                shots=1024
            )
            hx = torch.tensor(sampled, device=outputs.device, dtype=outputs.dtype)
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

class QLayer(tq.QuantumModule):
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.size(0), device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        return self.measure(qdev)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(tq.QuantumModule):
    def __init__(self, embed_dim, ffn_dim, n_qubits, dropout=0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.size(0), device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        q_out = self.measure(qdev)
        q_out = self.linear1(self.dropout(q_out))
        return self.linear2(F.relu(q_out))

class MultiHeadAttentionQuantum(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, q_device=None):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_device = q_device or tq.QuantumDevice(n_wires=embed_dim)
        self.q_layer = QLayer(embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        qdev = self.q_device.copy(bsz=batch_size, device=x.device)
        q_out = self.q_layer(x.view(-1, embed_dim), qdev)
        q_out = q_out.view(batch_size, seq_len, embed_dim)
        q = k = v = q_out
        scores = torch.matmul(q, k.transpose(-2,-1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        attn = torch.matmul(scores, v)
        return attn

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim,
                 n_qubits_transformer=0, n_qubits_ffn=0,
                 n_qlayers=1, q_device=None, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
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

class QuantumSelfAttention:
    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
        self.qc = tq.QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self.qc.rx(tq.Parameter(f"rx{i}"), i)
            self.qc.ry(tq.Parameter(f"ry{i}"), i)
            self.qc.rz(tq.Parameter(f"rz{i}"), i)
        for i in range(n_qubits-1):
            self.qc.crx(tq.Parameter(f"crx{i}"), i, i+1)
        self.qc.measure_all()

    def run(self, backend, rotation_params, entangle_params, inputs, shots=1024):
        results = []
        for inp in inputs:
            circ = self.qc.copy()
            for i in range(self.n_qubits):
                circ.rx(rotation_params[3*i], i)
                circ.ry(rotation_params[3*i+1], i)
                circ.rz(rotation_params[3*i+2], i)
            for i in range(self.n_qubits-1):
                circ.crx(entangle_params[i], i, i+1)
            job = tq.execute(circ, backend, shots=shots)
            results.append(job.result().get_counts(circ))
        return results

class QuantumSamplerQNN:
    def __init__(self):
        self.qc = tq.QuantumCircuit(2)
        self.qc.ry(tq.Parameter("input0"), 0)
        self.qc.ry(tq.Parameter("input1"), 1)
        self.qc.cx(0,1)
        self.qc.ry(tq.Parameter("weight0"), 0)
        self.qc.ry(tq.Parameter("weight1"), 1)
        self.qc.cx(0,1)
        self.qc.ry(tq.Parameter("weight2"), 0)
        self.qc.ry(tq.Parameter("weight3"), 1)
        self.sampler = tq.StatevectorSampler()
        self.qnn = tq.SamplerQNN(circuit=self.qc,
                                 input_params=[tq.Parameter("input0"), tq.Parameter("input1")],
                                 weight_params=[tq.Parameter("weight0"), tq.Parameter("weight1"),
                                                tq.Parameter("weight2"), tq.Parameter("weight3")],
                                 sampler=self.sampler)

    def run(self, backend, inputs, shots=1024):
        return self.qnn(inputs=inputs, shots=shots)

__all__ = ["QLSTM", "LSTMTagger"]
