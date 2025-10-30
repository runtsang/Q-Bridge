"""Quantum hybrid classifier mirroring :class:`HybridKernelClassifier`.

This module uses quantum convolution, quantum self‑attention and a quantum kernel
evaluated on a parameterised circuit.  The implementation follows the same
API as the classical version but relies on TorchQuantum and Qiskit for the
quantum components.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit

# Quantum helpers from the seed modules
from Conv import Conv as QuantumConv
from SelfAttention import SelfAttention as QuantumSelfAttention
from QuantumKernelMethod import KernalAnsatz as QuantumKernalAnsatz, Kernel as QuantumKernel

class HybridKernelClassifier(nn.Module):
    """Quantum‑classical hybrid classifier that fuses quantum convolution,
    quantum self‑attention and a quantum kernel."""
    def __init__(self,
                 n_qubits: int = 4,
                 backend=None,
                 shots: int = 1024,
                 rbf_gamma: float = 1.0,
                 num_classes: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Quantum convolution
        self.qconv = QuantumConv()
        # Quantum self‑attention
        self.qattn = QuantumSelfAttention()
        # Quantum kernel
        self.qkernel = QuantumKernel()
        # Dense head
        self.fc = nn.Linear(in_features=n_qubits + 1, out_features=num_classes)

        # Prototype vector for kernel evaluation
        self.register_buffer('prototype', torch.randn((1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Logits for each class.
        """
        batch = x.shape[0]

        # Quantum convolution: produce a scalar per sample
        conv_vals = []
        for i in range(batch):
            arr = x[i].squeeze().cpu().numpy()
            conv_vals.append(self.qconv.run(arr))
        conv_tensor = torch.tensor(conv_vals, device=x.device).unsqueeze(-1)  # shape (batch, 1)

        # Quantum self‑attention: produce a vector of size n_qubits
        attn_vals = []
        for i in range(batch):
            rotation = np.eye(self.n_qubits, dtype=np.float32)
            entangle = np.eye(self.n_qubits, dtype=np.float32)
            counts = self.qattn.run(
                backend=self.backend,
                rotation_params=rotation,
                entangle_params=entangle,
                shots=self.shots
            )
            # Convert counts dictionary to a probability vector over qubits
            probs = np.zeros(self.n_qubits)
            total = sum(counts.values())
            for bitstring, cnt in counts.items():
                for idx, bit in enumerate(bitstring[::-1]):  # reverse to match qubit order
                    probs[idx] += cnt * int(bit)
            probs /= total
            attn_vals.append(probs)
        attn_tensor = torch.tensor(attn_vals, device=x.device)  # shape (batch, n_qubits)

        # Quantum kernel: expectation value between conv output and prototype
        kernel_vals = []
        for i in range(batch):
            val = self.qkernel(conv_tensor[i], self.prototype[i])
            kernel_vals.append(val.item())
        kernel_tensor = torch.tensor(kernel_vals, device=x.device).unsqueeze(-1)  # shape (batch, 1)

        # Concatenate attention and kernel features
        features = torch.cat([attn_tensor, kernel_tensor], dim=-1)  # shape (batch, n_qubits+1)

        # Dense head
        logits = self.fc(features)
        return F.softmax(logits, dim=-1)

__all__ = ["HybridKernelClassifier"]
