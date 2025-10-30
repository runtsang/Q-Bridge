"""Quantum self‑attention model built with Pennylane and PyTorch.

Combines a convolutional feature extractor (QuantumNAT style) with a
parameterised quantum circuit that implements the attention mechanism.
"""

import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as pnp


class SelfAttentionHybrid(nn.Module):
    """
    Quantum‑based self‑attention module.
    """
    def __init__(self, in_channels: int = 1, n_wires: int = 4, shots: int = 1024):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.qkv_proj = nn.Linear(16 * 7 * 7, 3 * n_wires)
        self.n_wires = n_wires
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=self.n_wires)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def attention_circuit(params: torch.Tensor, x: torch.Tensor):
            # Encode parameters and input into rotations
            for i in range(self.n_wires):
                qml.RX(params[i], wires=i)
                qml.RZ(x[i], wires=i)
            # Entangling layer
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measurement: expectation of PauliZ on each wire
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_wires)]

        self.attention_circuit = attention_circuit
        self.out_proj = nn.Linear(self.n_wires, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f_flat = self.flatten(f)
        qkv = self.qkv_proj(f_flat)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        bsz = x.size(0)
        out_list = []
        for i in range(bsz):
            # Prepare parameters and input vector for the circuit
            params = torch.randn(self.n_wires, device=x.device, requires_grad=True)
            inp = (q[i] + k[i])[:self.n_wires]
            logits = self.attention_circuit(params, inp)
            logits = torch.tensor(logits, device=x.device)
            attn = torch.softmax(logits, dim=0)
            out = attn * v[i, :self.n_wires]
            out_list.append(out)
        out = torch.stack(out_list)
        out = self.out_proj(out)
        return self.norm(out)


def SelfAttention() -> SelfAttentionHybrid:
    """Factory returning an instance."""
    return SelfAttentionHybrid()


__all__ = ["SelfAttentionHybrid", "SelfAttention"]
