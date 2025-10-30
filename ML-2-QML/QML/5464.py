"""Quantum‑enhanced EstimatorQNN.

The quantum variant replaces the transformer blocks with TorchQuantum
modules and the output head with a hybrid expectation layer built on
Qiskit.  The public API matches the classical counterpart, so the
model can be swapped by simply importing this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter

# ---------- Autoencoder ----------
class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def Autoencoder(input_dim: int, *,
                latent_dim: int = 32,
                hidden_dims: tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    return AutoencoderNet(AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout))

# ---------- Quantum transformer components ----------
class QuantumAttentionBlock(nn.Module):
    """Simple quantum‑enhanced attention that maps each token through
    a small quantum circuit and uses the measurement results as
    attention scores.  The implementation follows the style of
    TorchQuantum's QLayer in the reference pair.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 q_device: tq.QuantumDevice | None = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        self.q_layer = self.QLayer(self.head_dim)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        outputs = []
        for head in x.unbind(dim=1):
            head_out = []
            for token in head.unbind(dim=1):
                qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
                head_out.append(self.q_layer(token, qdev))
            head_out = torch.stack(head_out, dim=1)
            outputs.append(head_out)
        out = torch.stack(outputs, dim=1)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.dropout(out)

class QuantumFeedForward(nn.Module):
    """Quantum feed‑forward block that uses a parameterised circuit to
    transform each token.  It follows the structure of the
    FeedForwardQuantum class from the reference pair.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
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

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    """Transformer block that replaces both attention and feed‑forward
    with quantum‑enabled modules.  The block is fully differentiable
    thanks to TorchQuantum's autograd support.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_attention: int = 8, n_qubits_ffn: int = 8,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttentionBlock(embed_dim, num_heads, dropout)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# ---------- Hybrid quantum head ----------
class QuantumCircuit:
    """Parametrised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: list[float]) -> float:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # expectation of Z for the first qubit
        counts = result
        if isinstance(counts, dict):
            probs = {k: v / self.shots for k, v in counts.items()}
            exp = sum((int(k[-1]) * 2 - 1) * p for k, p in probs.items())
            return exp
        return 0.0

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor,
                circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        exp_z = ctx.circuit.run(inputs.tolist())
        result = torch.tensor([exp_z], dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.tolist():
            exp_plus = ctx.circuit.run([val + shift])
            exp_minus = ctx.circuit.run([val - shift])
            grads.append(exp_plus - exp_minus)
        grad = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = []
        for val in inputs.squeeze(-1):
            outputs.append(HybridFunction.apply(val, self.circuit, self.shift))
        return torch.stack(outputs, dim=0).unsqueeze(-1)

# ---------- EstimatorQNN ----------
class EstimatorQNN(nn.Module):
    """Quantum‑enhanced estimator that mirrors the public API of the
    classical counterpart.  The model consists of an autoencoder,
    a stack of quantum transformer blocks, and a hybrid expectation
    head implemented with Qiskit.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 num_heads: int = 4,
                 num_blocks: int = 2,
                 ffn_dim: int = 128,
                 dropout: float = 0.1,
                 n_qubits_attention: int = 8,
                 n_qubits_ffn: int = 8) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(input_dim, latent_dim=latent_dim)
        self.pos_encoder = PositionalEncoder(latent_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockQuantum(latent_dim, num_heads, ffn_dim,
                                      n_qubits_attention, n_qubits_ffn, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(1, backend, shots=200, shift=3.1415926535 / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.autoencoder.encode(inputs)
        seq = encoded.unsqueeze(1)
        seq = self.pos_encoder(seq)
        x = self.transformer(seq)
        x = self.dropout(x.mean(dim=1, keepdim=True))
        return self.hybrid(x)

__all__ = ["EstimatorQNN", "Autoencoder", "AutoencoderNet", "AutoencoderConfig"]
