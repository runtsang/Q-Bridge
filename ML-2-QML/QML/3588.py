from __future__ import annotations

import math
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

# ---------------------------------- Quantum attention & feed‑forward ---------------------------------- #
class MultiHeadAttentionQuantum(tq.QuantumModule):
    """
    Multi‑head attention realised as a stack of small parameterised quantum circuits.
    Each head applies a 2‑qubit circuit with 3 free angles.
    """
    class QHead(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 2
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                ]
            )
            self.params = nn.Parameter(torch.randn(3))

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            a, b, c = self.params
            q_device.rz(a, 0)
            q_device.ry(b, 1)
            q_device.cx(0, 1)
            q_device.rz(c, 1)
            return tq.MeasureAll(tq.PauliZ)(q_device)

    def __init__(self, embed_dim: int, num_heads: int, n_qlayers: int = 1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_heads = nn.ModuleList([self.QHead() for _ in range(num_heads)])
        self.n_qlayers = n_qlayers
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, seq, embed_dim)
        """
        batch, seq, _ = x.size()
        # split heads
        x = x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        out_heads = []
        for i, head in enumerate(self.q_heads):
            head_out = []
            for token in x[:, i, :, :].unbind(dim=1):  # iterate over sequence
                q_device = tq.QuantumDevice(n_wires=self.head_dim, bsz=token.size(0))
                head_out.append(head(token, q_device))
            out_heads.append(torch.stack(head_out, dim=1))
        out = torch.stack(out_heads, dim=1).transpose(1, 2).contiguous()
        # project back to embed_dim
        return self.combine(out.view(batch, seq, self.embed_dim))


class FeedForwardQuantum(tq.QuantumModule):
    """
    Two‑layer feed‑forward realised with a small quantum circuit.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.Parameter(torch.randn(n_wires))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, angle in enumerate(self.params):
                q_device.rx(angle, wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.q_layer = self.QLayer(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, seq, embed_dim)
        """
        batch, seq, _ = x.size()
        out = []
        for token in x.unbind(dim=1):  # iterate over sequence
            q_device = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=token.size(0))
            out.append(self.q_layer(token, q_device))
        out = torch.stack(out, dim=1)
        out = self.linear1(out)
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(tq.QuantumModule):
    """
    Quantum transformer block that chains attention and feed‑forward quantum modules.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qlayers: int = 1,
        n_q_heads_wires: int = 2,
        n_q_ffn_wires: int = 4,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, n_qlayers)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_q_ffn_wires)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ---------------------------------- QCNN ansatz ---------------------------------- #
class ZFeatureMap(tq.QuantumModule):
    """
    Simple feature‑map that rotates each qubit around X by the input value.
    """
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        return self.measure(q_device)


class ConvLayer(tq.QuantumModule):
    """
    2‑qubit convolution block with 3 parameterised rotations.
    """
    def __init__(self, num_qubits: int, n_qlayers: int = 1) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.n_qlayers = n_qlayers
        self.params = nn.Parameter(torch.randn(n_qlayers * num_qubits // 2 * 3))

    def forward(self, _x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        idx = 0
        for _ in range(self.n_qlayers):
            for q1, q2 in zip(range(0, self.num_qubits, 2), range(1, self.num_qubits, 2)):
                a, b, c = self.params[idx:idx + 3]
                q_device.rz(-math.pi / 2, q2)
                q_device.cx(q2, q1)
                q_device.rz(a, q1)
                q_device.ry(b, q2)
                q_device.cx(q1, q2)
                q_device.ry(c, q2)
                q_device.cx(q2, q1)
                q_device.rz(math.pi / 2, q1)
                idx += 3
        return tq.MeasureAll(tq.PauliZ)(q_device)


class PoolLayer(tq.QuantumModule):
    """
    2‑qubit pooling block that discards one qubit via a parametrised circuit.
    """
    def __init__(self, num_qubits: int, n_qlayers: int = 1) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.n_qlayers = n_qlayers
        self.params = nn.Parameter(torch.randn(n_qlayers * num_qubits // 2 * 3))

    def forward(self, _x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        idx = 0
        for _ in range(self.n_qlayers):
            for src, sink in zip(range(0, self.num_qubits, 2), range(1, self.num_qubits, 2)):
                a, b, c = self.params[idx:idx + 3]
                q_device.rz(-math.pi / 2, sink)
                q_device.cx(sink, src)
                q_device.rz(a, src)
                q_device.ry(b, sink)
                q_device.cx(src, sink)
                q_device.ry(c, sink)
                idx += 3
        return tq.MeasureAll(tq.PauliZ)(q_device)


# ---------------------------------- Full QCNN + Transformer quantum model ---------------------------------- #
class QCNNQuantumTransformer(tq.QuantumModule):
    """
    Quantum circuit that stitches together:
      • a Z‑feature map
      • a QCNN ansatz (conv → pool → conv → pool → conv → pool)
      • a stack of TransformerBlockQuantum layers
    The forward pass returns the expectation value of Pauli‑Z on all qubits.
    """
    def __init__(
        self,
        n_qubits: int = 8,
        n_transformer_blocks: int = 2,
        n_qlayers: int = 1,
        n_q_heads_wires: int = 2,
        n_q_ffn_wires: int = 4,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.feature_map = ZFeatureMap(n_qubits)

        # QCNN ansatz
        self.conv1 = ConvLayer(n_qubits, n_qlayers)
        self.pool1 = PoolLayer(n_qubits, n_qlayers)
        self.conv2 = ConvLayer(n_qubits // 2, n_qlayers)
        self.pool2 = PoolLayer(n_qubits // 2, n_qlayers)
        self.conv3 = ConvLayer(n_qubits // 4, n_qlayers)
        self.pool3 = PoolLayer(n_qubits // 4, n_qlayers)

        # Quantum transformer stack
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlockQuantum(
                    embed_dim=n_qubits // 4,
                    num_heads=2,
                    ffn_dim=16,
                    n_qlayers=n_qlayers,
                    n_q_heads_wires=n_q_heads_wires,
                    n_q_ffn_wires=n_q_ffn_wires,
                )
                for _ in range(n_transformer_blocks)
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, n_qubits)
        Returns expectation values of Z on all qubits.
        """
        q_device = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.size(0))
        # Feature map
        self.feature_map(x, q_device)

        # QCNN ansatz
        self.conv1(x, q_device)
        self.pool1(x, q_device)
        self.conv2(x, q_device)
        self.pool2(x, q_device)
        self.conv3(x, q_device)
        self.pool3(x, q_device)

        # Transformer layers
        for block in self.transformer_blocks:
            block(x, q_device)

        return self.measure(q_device)
