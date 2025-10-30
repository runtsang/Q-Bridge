"""Quantum‑enhanced transformer that mirrors the classical
`HybridTransformer`.  It replaces the transformer blocks with
quantum‑aware implementations and supplies a variational
classifier circuit.

Key innovations:
* A `QFCModel` based on the Quantum‑NAT example that produces a 4‑dimensional
  feature vector via a quantum circuit with a RandomLayer and custom gates.
* The transformer blocks use `MultiHeadAttentionQuantum` and
  `FeedForwardQuantum` which map linear projections through small
  quantum circuits.
* The classifier is a variational circuit built by
  `build_classifier_circuit_quantum` (from `QuantumClassifierModel.py`);
  its parameters are explicitly exposed to facilitate hybrid training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# ----------------------------------------------------------------------
# Quantum transformer primitives – adapted from the seed QTransformerTorch
# ----------------------------------------------------------------------
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding (identical to the classical version)."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: torch.Tensor | None = None

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, batch_size: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, self.attn_weights = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Attention where each linear projection is replaced by a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Simple RX encoding per wire
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            # entanglement pattern
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: tq.QuantumDevice | None = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self.QLayer()
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError(f"Input embedding {embed_dim} mismatch {self.embed_dim}")
        # Apply the same quantum layer to each token
        proj = self._apply_quantum_heads(x)
        out = self.downstream(proj, proj, proj, batch_size, mask)
        return self.combine(out)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = tq.QuantumDevice(
                    n_wires=self.q_layer.n_wires, bsz=head.size(0), device=head.device
                )
                head_outputs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1)

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardQuantum(FeedForwardBase):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int = 4) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 4, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(
        self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 4, q_device: tq.QuantumDevice | None = None, dropout: float = 0.1
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ----------------------------------------------------------------------
# Optional quantum image feature extractor – from QuantumNAT
# ----------------------------------------------------------------------
class QFCModel(tq.QuantumModule):
    """Quantum variant of the CNN‑FC feature extractor."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

# ----------------------------------------------------------------------
# Variational classifier circuit – from QuantumClassifierModel
# ----------------------------------------------------------------------
def build_classifier_circuit_quantum(
    num_qubits: int, depth: int
) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    """
    Construct a depth‑controlled variational circuit for classification.
    The circuit is identical to the one in the original seed but is
    returned together with the symbolic parameters for easy integration
    into hybrid training loops.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# ----------------------------------------------------------------------
# HybridTransformer – quantum variant
# ----------------------------------------------------------------------
class HybridTransformer(nn.Module):
    """
    Quantum‑enhanced transformer that mirrors the classical
    `HybridTransformer`.  It replaces the transformer blocks with
    quantum‑aware implementations and supplies a variational
    classifier circuit.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary for text inputs.
    embed_dim : int
        Dimensionality of token embeddings – also the number of qubits
        used in the classifier circuit.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer layers.
    ffn_dim : int
        Hidden dimensionality inside the MLP.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Drop‑out probability.
    classifier_depth : int, optional
        Depth of the variational classifier circuit.
    use_image_features : bool, optional
        If ``True`` the input is expected to be an image tensor and is
        processed by the quantum `QFCModel`.  The resulting 4‑dimensional
        vector is projected to ``embed_dim`` before feeding the transformer.
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
        classifier_depth: int = 2,
        use_image_features: bool = False,
    ) -> None:
        super().__init__()
        self.use_image_features = use_image_features
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        # Transformer stack with quantum blocks
        self.transformers = nn.Sequential(
            *[
                TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim, n_qubits_transformer=embed_dim,
                    n_qubits_ffn=embed_dim, dropout=dropout
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        # Optional quantum image encoder
        if use_image_features:
            self.image_extractor = QFCModel()
            self.image_embedding = nn.Linear(4, embed_dim)

        # Variational classifier circuit
        self.classifier_circuit, self.encoding, self.weights, self.observables = build_classifier_circuit_quantum(
            embed_dim, classifier_depth
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass – identical to the classical variant but the transformer
        blocks are quantum.  The output is the pooled transformer representation;
        classification is performed by the external quantum circuit.
        """
        if self.use_image_features:
            feats = self.image_extractor(x)                 # (B,4)
            tokens = self.image_embedding(feats).unsqueeze(1)  # (B,1,embed_dim)
            mask = None
        else:
            tokens = self.token_embedding(x)  # (B,seq_len,embed_dim)
            mask = None

        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))  # pooled representation
        return x  # raw features; to be fed into `self.classifier_circuit`

__all__ = [
    "PositionalEncoder",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "QFCModel",
    "build_classifier_circuit_quantum",
    "HybridTransformer",
]
