"""Hybrid transformer with autoencoder and hybrid classification head (quantum implementation)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import assemble, transpile
import numpy as np

# ----------------------------------------------------------------------
# Attention and feed‑forward primitives
# ----------------------------------------------------------------------
class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, v), scores

    def downstream(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch, -1, self.embed_dim)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        out = self.downstream(q, k, v, batch, mask)
        return self.combine_heads(out)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention where each projection is processed by a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        """Quantum layer that applies a parameterised rotation to each qubit."""
        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for gate, wire in zip(self.params, range(self.n_wires)):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 8) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.n_wires = n_wires
        self.q_layer = self.QLayer(n_wires)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum layer to each token projection."""
        batch, seq, _ = x.shape
        proj = x.view(batch, seq, self.num_heads, self.d_k)
        out = torch.zeros_like(proj)
        for b in range(batch):
            for s in range(seq):
                token = proj[b, s].reshape(self.num_heads, self.d_k)
                heads = []
                for h in range(self.num_heads):
                    head = token[h].unsqueeze(0)  # shape (1, d_k)
                    qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device=head.device)
                    heads.append(self.q_layer(head, qdev))
                heads = torch.stack(heads, dim=0)
                out[b, s] = heads
        return out.view(batch, seq, self.embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        batch, seq, _ = x.shape
        k = self._apply_quantum_heads(x)
        q = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        out = self.downstream(q, k, v, batch, mask)
        return self.combine_heads(out)

class FeedForwardBase(nn.Module):
    """Base class for feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward realised by a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for gate, wire in zip(self.params, range(self.n_wires)):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_wires)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch, seq, _ = x.shape
        out = torch.zeros(batch, seq, self.linear1.out_features, device=x.device)
        for b in range(batch):
            for s in range(seq):
                token = x[b, s].unsqueeze(0)  # shape (1, embed_dim)
                qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=1, device=token.device)
                qout = self.q_layer(token, qdev)
                out[b, s] = self.linear1(qout)
        out = self.linear2(self.dropout(F.relu(out)))
        return out

# ----------------------------------------------------------------------
# Transformer block
# ----------------------------------------------------------------------
class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that uses quantum attention and optionally quantum feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_wires_attn: int = 8, n_wires_ffn: int = 8, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires_attn)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_wires_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ----------------------------------------------------------------------
# Positional encoding
# ----------------------------------------------------------------------
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]

# ----------------------------------------------------------------------
# Autoencoder (classical)
# ----------------------------------------------------------------------
class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder for embedding reconstruction."""
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# ----------------------------------------------------------------------
# Quantum hybrid head
# ----------------------------------------------------------------------
class QuantumCircuitWrapper:
    """Simple parameterised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int = 1024) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridQuantumHead(nn.Module):
    """Hybrid layer that forwards activations through a Qiskit circuit."""
    def __init__(self, n_qubits: int, backend, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, features)
        thetas = x.detach().cpu().numpy()
        expectations = self.circuit.run(thetas)
        return torch.tensor(expectations, device=x.device)

# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class HybridTransformerAutoencoder(nn.Module):
    """Hybrid transformer that uses quantum attention/FFN and a quantum hybrid head."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        autoencoder_latent: int = 32,
        autoencoder_hidden: Tuple[int, int] = (128, 64),
        quantum_wires_attn: int = 8,
        quantum_wires_ffn: int = 8,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, quantum_wires_attn, quantum_wires_ffn, dropout)
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.autoencoder = AutoencoderNet(embed_dim, autoencoder_latent, autoencoder_hidden, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid_head = HybridQuantumHead(embed_dim, backend, shots=512, shift=shift)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            LongTensor of shape (batch, seq_len) containing token indices.

        Returns
        -------
        logits : torch.Tensor
            Classification logits of shape (batch, num_classes) or (batch, 1).
        reconstruction : torch.Tensor
            Reconstructed embeddings of shape (batch, seq_len, embed_dim).
        """
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.dropout(x)
        # Classification
        cls_logits = self.classifier(x.mean(dim=1))
        cls_hybrid = self.hybrid_head(x.mean(dim=1))
        logits = torch.cat((cls_logits, cls_hybrid), dim=-1) if self.classifier.out_features > 1 else cls_hybrid
        # Reconstruction
        reconstruction = self.autoencoder(x)
        return logits, reconstruction

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "AutoencoderNet",
    "HybridQuantumHead",
    "HybridTransformerAutoencoder",
]
