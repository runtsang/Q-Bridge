import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import qiskit
from qiskit import assemble, transpile
import math
from typing import Optional

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class MultiHeadAttentionQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

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

        self.qlayer = QLayer()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        def quantum_project(tensor):
            projections = []
            for token in tensor.unbind(dim=1):
                token = token.view(token.size(0), self.num_heads, -1)
                head_outputs = []
                for head in token.unbind(dim=1):
                    qdev = tq.QuantumDevice(n_wires=self.qlayer.n_wires, bsz=head.size(0), device=head.device)
                    head_outputs.append(self.qlayer(head, qdev))
                projections.append(torch.stack(head_outputs, dim=1))
            return torch.stack(projections, dim=1)

        k = quantum_project(k)
        q = quantum_project(q)
        v = quantum_project(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.combine_heads(out)

class FeedForwardQuantum(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        class QLayer(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = n_qubits
                self.encoder = tq.GeneralEncoder(
                    [
                        {"input_idx": [i], "func": "rx", "wires": [i]}
                        for i in range(n_qubits)
                    ]
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

        self.qlayer = QLayer()
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.qlayer.n_wires, bsz=token.size(0), device=token.device)
            outputs.append(self.qlayer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_transformer: int, n_qubits_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class QuantumCircuit:
    def __init__(self, n_qubits: int, backend, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, angles: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: angle} for angle in angles],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            probs = counts / self.shots
            exp = probs.get('0', 0) - probs.get('1', 0)
            return exp
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridQuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.circuit = circuit
        ctx.shift = shift
        angles = inputs.detach().cpu().numpy()
        exp_val = ctx.circuit.run(angles)
        out = torch.tensor(exp_val, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.cpu().numpy()) * ctx.shift
        grads = []
        for idx, val in enumerate(inputs.cpu().numpy()):
            right = ctx.circuit.run(np.array([val + shift[idx]]))
            left = ctx.circuit.run(np.array([val - shift[idx]]))
            grads.append(right[0] - left[0])
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class HybridQuantum(nn.Module):
    def __init__(self, in_features: int, backend, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        exp = HybridQuantumFunction.apply(inputs.squeeze(), self.circuit, self.shift)
        return torch.sigmoid(exp)

class HybridTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 8,
        n_qubits_head: int = 1,
        backend=None,
        shots: int = 1024,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.transformers = nn.Sequential(
            *[
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer,
                    n_qubits_ffn,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        if num_classes == 2:
            self.head = HybridQuantum(1, self.backend, shots=shots, shift=shift)
        else:
            self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        if hasattr(self, "head"):
            scalar = x.mean(dim=1)
            probs = self.head(scalar)
            return torch.cat((probs, 1 - probs), dim=-1)
        else:
            return self.classifier(x)

__all__ = ["HybridTransformerClassifier"]
