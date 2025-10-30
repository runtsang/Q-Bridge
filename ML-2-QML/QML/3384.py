"""Hybrid text classifier with a quantum convolutional front‑end and
optional quantum transformer blocks.

The quantum implementation mirrors the classical version but replaces
the convolutional filter with a parameterised circuit (QuanvCircuit)
and allows the transformer layers to be fully quantum or classical
based on boolean flags.  The API is identical to the classical
HybridTextClassifier, enabling seamless experimentation with
hybrid architectures.

Author: gpt-oss-20b
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit.circuit.random import random_circuit
import numpy as np


# --------------------------------------------------------------------------- #
# Quantum convolutional filter (quanvolution)
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Quantum filter circuit used for the quanvolution front‑end.

    The circuit prepares a set of qubits with RX gates whose angles
    depend on the input pixel values, followed by a small random
    circuit and measurement in the computational basis.  The output
    is the average probability of measuring |1> across all qubits.
    """
    def __init__(self, kernel_size: int, backend: qiskit.providers.BaseBackend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit on a single 2‑D patch.

        Parameters
        ----------
        data : np.ndarray
            Patch of shape (kernel_size, kernel_size) with values in [0, 255].

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


# --------------------------------------------------------------------------- #
# Classical convolutional filter (fallback)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Pure‑PyTorch 2‑D convolutional filter that outputs a scalar per patch.

    This class is identical to the one in the classical implementation
    and is used when the user opts out of the quantum front‑end.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            padding=0,
        )
        logits = self.conv(x).view(x.size(0), 1, -1, self.kernel_size, self.kernel_size)
        logits = logits.mean(dim=[3, 4])
        activations = torch.sigmoid(logits - self.threshold)
        return activations


# --------------------------------------------------------------------------- #
# Classical transformer components (fallback)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# --------------------------------------------------------------------------- #
# Quantum transformer components
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(nn.Module):
    """Quantum‑enhanced multi‑head attention.

    Each head is implemented by a small quantum circuit that maps
    the input embeddings to a new representation.  The circuit
    consists of RX encoders, a CNOT ladder, and a measurement in
    the Pauli‑Z basis.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
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

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum circuit to each head of each token."""
        batch_size = x.size(0)
        seq_len = x.size(1)
        head_dim = self.d_k
        outputs = []
        for token in x.unbind(dim=1):
            token = token.view(batch_size, self.num_heads, head_dim)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=token.size(0), device=head.device)
                head_outputs.append(self.q_layer(head, qdev))
            outputs.append(torch.stack(head_outputs, dim=1))
        return torch.stack(outputs, dim=1)  # (B, seq_len, num_heads, head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self._apply_quantum_heads(x)
        # Classical attention on the quantum outputs
        batch_size, seq_len, num_heads, head_dim = q.size()
        q = q.view(batch_size, seq_len, self.embed_dim)
        k = q
        v = q
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        return self.combine_heads(out)


class FeedForwardQuantum(nn.Module):
    """Quantum feed‑forward network realized by a parameterised circuit."""
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
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    """Transformer block that can mix classical and quantum sub‑modules."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        use_quantum_attention: bool = True,
        use_quantum_ffn: bool = True,
        n_qubits_ffn: int = 8,
        q_device: Optional[tq.QuantumDevice] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        if use_quantum_attention:
            self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        if use_quantum_ffn:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
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


# --------------------------------------------------------------------------- #
# Hybrid classifier (quantum front‑end + optional quantum transformer)
# --------------------------------------------------------------------------- #
class HybridTextClassifier(nn.Module):
    """Hybrid transformer classifier with a quantum convolutional front‑end.

    Parameters
    ----------
    image_size : int
        Height/width of the square input image.
    patch_size : int
        Size of the square patches extracted by the quanvolution filter.
    embed_dim : int
        Dimensionality of the transformer embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Feed‑forward hidden dimension.
    num_classes : int
        Number of target classes.
    dropout : float, optional
        Drop‑out probability.
    use_quantum_conv : bool, optional
        Whether to use the quantum convolutional filter.  If False,
        a classical ConvFilter is used.
    use_quantum_attention : bool, optional
        Whether to use quantum attention in the transformer blocks.
    use_quantum_ffn : bool, optional
        Whether to use quantum feed‑forward networks in the transformer blocks.
    n_qubits_ffn : int, optional
        Number of qubits for the quantum feed‑forward circuit.
    """
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum_conv: bool = True,
        use_quantum_attention: bool = True,
        use_quantum_ffn: bool = True,
        n_qubits_ffn: int = 8,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2

        # Convolutional front‑end
        if use_quantum_conv:
            backend = qiskit.Aer.get_backend("qasm_simulator")
            self.conv_filter = QuanvCircuit(
                kernel_size=patch_size,
                backend=backend,
                shots=100,
                threshold=127,
            )
        else:
            self.conv_filter = ConvFilter(kernel_size=patch_size)

        # Linear projection from scalar patch activation to embedding space
        self.patch_embed = nn.Linear(1, embed_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoder(embed_dim)

        # Transformer encoder
        self.transformer = nn.Sequential(
            *[
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    use_quantum_attention=use_quantum_attention,
                    use_quantum_ffn=use_quantum_ffn,
                    n_qubits_ffn=n_qubits_ffn,
                    q_device=None,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim,
            num_classes if num_classes > 2 else 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, H, W) or (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes) or (B, 1) for binary.
        """
        # Convolutional filtering → (B, 1, N_patches)
        if isinstance(self.conv_filter, QuanvCircuit):
            # Convert torch tensor to numpy patches
            patches = x.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
            # patches shape: (B, num_patches_h, num_patches_w, patch_size, patch_size)
            patches = patches.contiguous().view(x.size(0), -1, self.patch_size, self.patch_size)
            activations = []
            for patch in patches:
                # patch shape: (B, patch_size, patch_size)
                batch_act = torch.stack(
                    [torch.tensor(self.conv_filter.run(p.cpu().numpy()), dtype=torch.float32) for p in patch]
                )
                activations.append(batch_act)
            activations = torch.stack(activations, dim=1)  # (B, N_patches, 1)
        else:
            activations = self.conv_filter(x)  # (B, 1, N_patches)

        # Project to embedding space
        x = self.patch_embed(activations)  # (B, N_patches, embed_dim)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)

        # Pool and classify
        x = x.mean(dim=1)  # global average pooling
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "QuanvCircuit",
    "ConvFilter",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTextClassifier",
]
