from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit.providers.aer import AerSimulator

# Autoencoder
class AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# Classical self‑attention
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)

# Quantum classifier circuit
class QuantumClassifierCircuit:
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.theta = [f"theta_{i}" for i in range(n_qubits)]
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits)
        cr = ClassicalRegister(self.n_qubits)
        qc = QuantumCircuit(qr, cr)
        for i in range(self.n_qubits):
            qc.h(qr[i])
        for i in range(self.n_qubits - 1):
            qc.cx(qr[i], qr[i + 1])
        qc.barrier()
        for i, param in enumerate(self.theta):
            qc.ry(f"{{{param}}}", qr[i])
        qc.measure(qr, cr)
        return qc

    def run(self, params: np.ndarray) -> np.ndarray:
        param_dict = {self.theta[i]: params[i] for i in range(len(params))}
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[param_dict])
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = 0.0
        for state, cnt in counts.items():
            bit = int(state[::-1][0])  # first qubit
            exp += ((-1) ** bit) * cnt
        exp /= self.shots
        return np.array([exp])

class QuantumHybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumClassifierCircuit):
        ctx.circuit = circuit
        inputs_np = inputs.detach().cpu().numpy()
        outputs = np.vstack([circuit.run(inp) for inp in inputs_np])
        outputs_t = torch.from_numpy(outputs).to(inputs.device).float()
        ctx.save_for_backward(inputs, outputs_t)
        return outputs_t

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        eps = 1e-3
        grads = []
        for i in range(inputs.size(1)):
            pert_plus = inputs.clone()
            pert_minus = inputs.clone()
            pert_plus[:, i] += eps
            pert_minus[:, i] -= eps
            out_plus = ctx.circuit.run(pert_plus.detach().cpu().numpy())
            out_minus = ctx.circuit.run(pert_minus.detach().cpu().numpy())
            grad = (out_plus - out_minus) / (2 * eps)
            grads.append(grad)
        grad_input = torch.from_numpy(np.vstack(grads).T).to(inputs.device).float()
        return grad_input * grad_output, None

class FraudDetectionHybrid(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 attention_dim: int = 4,
                 n_qubits: int = 4):
        super().__init__()
        self.autoencoder = AutoencoderNet(input_dim, latent_dim, hidden_dims, dropout)
        self.attention = ClassicalSelfAttention(attention_dim)
        self.attn_to_qubits = nn.Linear(attention_dim, n_qubits)
        self.quantum_circuit = QuantumClassifierCircuit(n_qubits)
        self._head = QuantumHybridFunction.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        if z.size(-1)!= self.attention.embed_dim:
            z = F.pad(z, (0, self.attention.embed_dim - z.size(-1)), mode='constant')
        attn_out = self.attention(z)
        qubit_params = self.attn_to_qubits(attn_out)
        logits = self._head(qubit_params, self.quantum_circuit)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["FraudDetectionHybrid"]
