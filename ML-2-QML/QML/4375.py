"""Hybrid transformer classifier with quantum attention, photonic feature extractor,
and a Qiskit quantum classifier head.

This module extends the classical implementation by replacing the
transformer blocks with their quantum counterparts from TorchQuantum,
adding a Strawberry Fields photonic fraud‑detection layer, and
providing a quantum classifier head that can be evaluated with
Qiskit’s SamplerQNN.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.quantum_info import SparsePauliOp

# Local imports – the seed modules are assumed to be on the PYTHONPATH
from QTransformerTorch import PositionalEncoder
from FraudDetection import build_fraud_detection_program
from QuantumClassifierModel import build_classifier_circuit


# --------------------------------------------------------------------------- #
#  Quantum transformer primitives (simplified copy of the seed)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, batch_size: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, self.attn_weights = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention that maps projections through quantum modules."""

    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(self.n_wires)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self.QLayer()
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError(f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})")
        k = self._apply_quantum_heads(x)
        q = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        x = self.downstream(q, k, v, batch_size, mask)
        return self.combine_heads(x)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=head.size(0), device=head.device)
                head_outputs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1)


class FeedForwardBase(nn.Module):
    """Shared interface for feed‑forward layers."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a quantum module."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_qubits)]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    """Quantum‑enhanced transformer block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_attention: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Photonic feature extractor (fraud‑detection style)
# --------------------------------------------------------------------------- #
class PhotonicFeatureExtractor(nn.Module):
    """
    Wraps a Strawberry‑Fields program that implements a photonic fraud‑detection
    circuit.  The program outputs a two‑dimensional feature vector that can be
    fed into the transformer.
    """

    def __init__(self, program: sf.Program, engine: Engine) -> None:
        super().__init__()
        self.program = program
        self.engine = engine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for the simulator
        inp = x.detach().cpu().numpy()
        results = self.engine.run(self.program, args=inp)
        # The program returns a tuple of two outputs per sample
        out = torch.tensor(results["samples"], device=x.device, dtype=x.dtype)
        return out


# --------------------------------------------------------------------------- #
#  Hybrid transformer classifier
# --------------------------------------------------------------------------- #
class HybridTransformerClassifier(nn.Module):
    """
    A transformer‑based classifier that can use either classical or
    quantum transformer blocks, prepend a photonic feature extractor,
    and append a Qiskit quantum classifier head.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        *,
        use_photonic: bool = False,
        photonic_program: Optional[sf.Program] = None,
        photonic_engine: Optional[Engine] = None,
        use_quantum_attention: bool = False,
        use_quantum_ffn: bool = False,
        n_qubits_attention: int = 8,
        n_qubits_ffn: int = 8,
        use_quantum_classifier: bool = False,
        qiskit_circuit_depth: int = 3,
        qiskit_num_qubits: int = 4,
    ) -> None:
        super().__init__()

        # Optional photonic feature extractor
        if use_photonic:
            if photonic_program is None or photonic_engine is None:
                raise ValueError("photonic_program and photonic_engine must be supplied when use_photonic is True")
            self.photonic = PhotonicFeatureExtractor(photonic_program, photonic_engine)
        else:
            self.photonic = None

        # Token embedding and positional encoding (only used for text inputs)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        # Transformer stack – choose classical or quantum blocks
        if use_quantum_attention or use_quantum_ffn:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_attention,
                        n_qubits_ffn,
                        dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            # Import the classical block from the seed
            from QTransformerTorch import TransformerBlockClassical

            self.transformers = nn.Sequential(
                *[
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

        # Optional quantum classifier head
        self.quantum_head: Optional[SamplerQNN] = None
        if use_quantum_classifier:
            circuit, enc_params, var_params, observables = build_classifier_circuit(
                qiskit_num_qubits, qiskit_circuit_depth
            )
            sampler = StatevectorSampler()
            self.quantum_head = SamplerQNN(
                circuit=circuit,
                weight_params=var_params,
                input_params=enc_params,
                interpret=lambda x: x,
                output_shape=len(observables),
                sampler=sampler,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.  If ``use_photonic`` is True, ``x`` is expected to be
            a continuous feature set of shape (batch, 2).  Otherwise it is a
            sequence of token indices for text classification.
        """
        # Photonic feature extraction
        if self.photonic is not None:
            x = self.photonic(x)

        # Token embedding and positional encoding (skip if photonic)
        if self.photonic is None:
            tokens = self.token_embedding(x)
            x = self.pos_embedding(tokens)

        # Transformer stack
        x = self.transformers(x)

        # Pooling, dropout, and classification head
        x = self.dropout(x.mean(dim=1))
        logits = self.classifier(x)

        # Quantum classifier head (if configured)
        if self.quantum_head is not None:
            q_input = logits.detach().cpu().numpy()
            q_out = self.quantum_head(q_input)
            logits = torch.tensor(q_out, device=x.device, dtype=x.dtype)

        return logits


__all__ = [
    "HybridTransformerClassifier",
]
