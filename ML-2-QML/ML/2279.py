"""Hybrid convolutional‑sampler module for classical training pipelines.

The class `ConvSamplerHybrid` is a drop‑in replacement for the original
`Conv` class.  It contains:
* a fast classical 2‑D convolution (`torch.nn.Conv2d`);
* a lightweight quantum‑convolution wrapper that runs a Qiskit circuit
  on the same kernel window;
* a small neural sampler (`SamplerQNN`) that produces class probabilities
  from the concatenated classical/quantum features.

The module is fully compatible with PyTorch training loops and can be
instantiated exactly as the original `Conv` class.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import qiskit
from qiskit.circuit import ParameterVector
from qiskit import Aer, execute
from qiskit.circuit.random import random_circuit

__all__ = ["ConvSamplerHybrid"]


class _QuantumConvWrapper(nn.Module):
    """Wraps a Qiskit circuit that emulates a convolutional filter."""

    def __init__(self, kernel_size: int = 2, shots: int = 200, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold

        # Build a parameterised circuit
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = ParameterVector("theta", self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

        self.backend = Aer.get_backend("qasm_simulator")

    def forward(self, kernel: torch.Tensor) -> torch.Tensor:
        # Convert kernel to 1‑D array of values
        data = kernel.detach().cpu().numpy().reshape(1, self.n_qubits)
        param_binds = []
        for row in data:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(row)}
            param_binds.append(bind)

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)
        # Compute average probability of measuring |1> over all qubits
        counts = 0
        for bitstring, freq in result.items():
            ones = sum(int(b) for b in bitstring)
            counts += ones * freq
        avg_prob = counts / (self.shots * self.n_qubits)
        return torch.tensor(avg_prob, dtype=torch.float32, device=kernel.device)


class SamplerQNN(nn.Module):
    """Small neural sampler that maps 2‑D inputs to class logits."""

    def __init__(self, in_features: int = 2, hidden: int = 4, out_features: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class ConvSamplerHybrid(nn.Module):
    """Hybrid convolutional‑sampler network."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Classical convolution
        self.classical_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Quantum convolution wrapper
        self.quantum_conv = _QuantumConvWrapper(kernel_size=kernel_size, threshold=threshold)

        # Sampler QNN
        self.sampler = SamplerQNN(in_features=2, hidden=4, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, H, W) where B is batch size.
        Returns:
            Tensor of shape (B, 2) with class probabilities.
        """
        # Classical feature map
        class_feat = self.classical_conv(x)  # (B, 1, H-k+1, W-k+1)
        class_feat = class_feat.view(x.size(0), -1)  # flatten

        # Quantum feature map
        # We apply the quantum conv to each spatial window
        batch, _, h, w = x.shape
        q_feats = []
        for b in range(batch):
            for i in range(0, h - self.kernel_size + 1, self.kernel_size):
                for j in range(0, w - self.kernel_size + 1, self.kernel_size):
                    patch = x[b, 0, i : i + self.kernel_size, j : j + self.kernel_size]
                    q_val = self.quantum_conv(patch)
                    q_feats.append(q_val)
        q_feats = torch.stack(q_feats).view(batch, -1)

        # Concatenate classical and quantum features
        features = torch.cat([class_feat, q_feats], dim=-1)

        # Pass through sampler
        probs = self.sampler(features)
        return probs
