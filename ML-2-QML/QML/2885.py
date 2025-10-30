import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit.circuit.random import random_circuit
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumConv:
    """
    Quantum filter implemented with a Qiskit circuit.
    Produces an average probability of measuring |1> across qubits.
    """
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: int = 127):
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
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

class PositionalEncoder(nn.Module):
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

class MultiHeadAttentionQuantum(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self):
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

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = (
                    self.q_device
                    or tq.QuantumDevice(
                        n_wires=self.q_layer.n_wires,
                        bsz=head.size(0),
                        device=head.device,
                    )
                )
                head_outputs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = t.size()
        return t.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        k = self._apply_quantum_heads(x)
        q = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(embed_dim // self.num_heads)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        attn = torch.matmul(scores, v)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.combine_heads(attn)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
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

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        n_qubits: int,
        dropout: float = 0.1,
    ) -> None:
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
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        n_qlayers: int,
        q_device: tq.QuantumDevice | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device)
        self.ffn = (
            FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
            if n_qubits_ffn > 0
            else FeedForwardClassical(embed_dim, ffn_dim, dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridTransformerConv(nn.Module):
    """
    Quantum‑enabled hybrid transformer.
    The constructor exposes quantum hyper‑parameters; if they are set to zero
    the module falls back to the classical implementation.
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
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        super().__init__()
        self.conv = QuantumConv()
        self.conv_linear = nn.Linear(1, embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        if n_qubits_transformer > 0:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_transformer,
                        n_qubits_ffn,
                        n_qlayers,
                        q_device,
                        dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input `x` is a tensor of shape (batch, H, W). The quantum filter is applied
        per sample, producing a scalar feature that is linearly projected into the
        transformer embedding space.
        """
        conv_vals = []
        for sample in x:
            conv_vals.append(self.conv.run(sample.cpu().numpy()))
        conv_tensor = torch.tensor(conv_vals, device=x.device).unsqueeze(1)  # (batch, 1)
        conv_emb = self.conv_linear(conv_tensor)
        x = conv_emb
        x = self.pos_embedding(x)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = ["HybridTransformerConv"]
