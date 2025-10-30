import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import qiskit
import torchquantum as tq
import torchquantum.functional as tqf


class FCLQuantum:
    """Quantum fully‑connected layer based on a parameterized variational circuit."""
    def __init__(self, n_qubits, backend, shots=100):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas):
        if len(thetas)!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} thetas, got {len(thetas)}")
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array([int(k, 2) for k in result.keys()])
        probs = counts / self.shots
        expectations = []
        for bit in range(self.n_qubits):
            bits = ((states >> bit) & 1)
            exp = np.sum((1 - 2 * bits) * probs)  # 0 -> +1, 1 -> -1
            expectations.append(exp)
        return np.array(expectations)  # shape (n_qubits,)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented classically."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)


class FeedForwardClassical(nn.Module):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class ClassicalTransformerBlock(nn.Module):
    """A single transformer block with classical attention and feed‑forward."""
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


class HybridTransformer(nn.Module):
    """
    Hybrid transformer that uses a quantum fully‑connected layer for the
    token embedding and classical transformer blocks for sequence modeling.
    """
    def __init__(self, config: dict):
        super().__init__()
        # Quantum fully‑connected layer
        self.fcl = FCLQuantum(
            n_qubits=config["embed_dim"],
            backend=qiskit.Aer.get_backend("qasm_simulator"),
            shots=100,
        )
        # Positional encoding
        self.pos_encoder = PositionalEncoder(config["embed_dim"])
        # Transformer blocks
        self.transformer = nn.Sequential(
            *[ClassicalTransformerBlock(
                config["embed_dim"],
                config["num_heads"],
                config["ffn_dim"],
                config.get("dropout", 0.1),
            ) for _ in range(config["num_blocks"])]
        )
        # Classifier
        num_classes = config["num_classes"]
        if num_classes > 2:
            self.classifier = nn.Linear(config["embed_dim"], num_classes)
        else:
            self.classifier = nn.Linear(config["embed_dim"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, n_features)
        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes) or (batch, 1)
        """
        batch, seq_len, n_features = x.shape
        # Compute quantum embedding for each token
        embeddings = []
        for i in range(seq_len):
            token = x[:, i, :]  # (batch, n_features)
            token_thetas = token.detach().cpu().numpy()
            token_embeds = []
            for theta_vec in token_thetas:
                token_embeds.append(self.fcl.run(theta_vec))
            token_embeds = np.stack(token_embeds, axis=0)  # (batch, embed_dim)
            embeddings.append(torch.tensor(token_embeds, dtype=torch.float32, device=x.device))
        x = torch.stack(embeddings, dim=1)  # (batch, seq_len, embed_dim)
        # Positional encoding
        x = self.pos_encoder(x)
        # Transformer
        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=1)
        # Classification
        return self.classifier(x)
