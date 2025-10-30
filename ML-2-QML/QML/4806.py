"""Hybrid classical‑quantum network with an autoencoder, self‑attention, and a variational quantum head.

The model mirrors the purely classical architecture but replaces the final
dense layer with a differentiable quantum expectation value, enabling
non‑classical feature transformations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

# ----- Classical building blocks (copied‑but‑restructured) -----
class AEConfig:
    def __init__(self, input_dim, latent_dim=32, hidden_dims=(128, 64), dropout=0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AEConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x).unsqueeze(2)  # (B, E, 1)
        k = self.key(x).unsqueeze(1)    # (B, 1, E)
        v = self.value(x).unsqueeze(2)  # (B, E, 1)
        scores = F.softmax(torch.bmm(q, k) / np.sqrt(self.embed_dim), dim=-1)
        return torch.bmm(scores, v).squeeze(2)


# ----- Quantum helper -----
class QuantumCircuitWrapper:
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")

        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in count_dict.keys()]).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.tolist())
        result = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        grad_inputs = []
        for val in inputs.numpy():
            right = ctx.circuit.run([val + shift[0]])
            left = ctx.circuit.run([val - shift[0]])
            grad_inputs.append(right - left)
        grad = torch.tensor(grad_inputs, dtype=torch.float32)
        return grad * grad_output, None, None


class Hybrid(nn.Module):
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)


# ----- Hybrid model -----
class HybridAutoEncoderAttentionNet(nn.Module):
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        # Convolutional backbone identical to the classical version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Feature size after convolution
        dummy = torch.zeros(1, 3, 32, 32)
        flat = self._forward_conv(dummy)
        flat_size = flat.shape[1]

        # Classical autoencoder
        ae_cfg = AEConfig(
            input_dim=flat_size,
            latent_dim=32,
            hidden_dims=(128, 64),
            dropout=0.1,
        )
        self.autoencoder = AutoencoderNet(ae_cfg)

        # Self‑attention on latent vector
        self.attention = SelfAttentionLayer(embed_dim=ae_cfg.latent_dim)

        # Quantum hybrid head
        backend = AerSimulator()
        self.hybrid = Hybrid(n_qubits=3, backend=backend, shots=512, shift=shift)
        self.shift = shift

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        latent = self.autoencoder.encode(x)
        attn_out = self.attention(latent)
        # Reduce latent to a single value for the quantum circuit
        latent_mean = attn_out.mean(dim=1)
        quantum_out = self.hybrid(latent_mean)
        probs = torch.sigmoid(quantum_out + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["AEConfig", "AutoencoderNet", "SelfAttentionLayer", "HybridAutoEncoderAttentionNet"]
