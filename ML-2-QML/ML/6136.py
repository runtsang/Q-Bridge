"""Hybrid classical‑quantum convolution module.

This module keeps the original drop‑in API but adds a quantum
feature extractor that can be enabled or disabled.  The classical
part uses a learnable 2‑D kernel, while the quantum part creates a
parameterised circuit that acts on the flattened patch.  The two
outputs are concatenated and returned as separate channels.
"""

import numpy as np
import torch
from torch import nn
from typing import Optional
import qiskit
from qiskit.circuit.random import random_circuit

class _QuantumFeatureExtractor:
    """Wraps a parameterised Qiskit circuit that acts on a flattened patch.
    The circuit is fixed after construction; only the parameters
    (θ) are set at runtime based on the input patch.
    """
    def __init__(self, kernel_size: int, backend: qiskit.providers.BaseBackend,
                 shots: int, threshold: float):
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

    def run(self, patch: np.ndarray) -> float:
        """Return average probability of measuring |1> across qubits."""
        data = np.reshape(patch, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(self._circuit, self.backend,
                             shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class ConvHybrid(nn.Module):
    """Hybrid convolutional filter that optionally uses a quantum
    feature extractor.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 use_quantum: bool = False,
                 quantum_backend: Optional[str] = None,
                 shots: int = 100,
                 threshold: float = 0.5,
                 bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias)
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.use_quantum = use_quantum
        self.quantum_backend = quantum_backend
        self.shots = shots
        self.quantum_extractor: Optional[_QuantumFeatureExtractor] = None

        if self.use_quantum:
            if quantum_backend is None:
                raise ValueError("quantum_backend must be specified when use_quantum=True")
            backend = qiskit.Aer.get_backend(quantum_backend)
            self.quantum_extractor = _QuantumFeatureExtractor(
                kernel_size, backend, shots, threshold
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, 1, H, W) where H and W are at least
               kernel_size. The module applies a 2‑D convolution to each
               channel and, if enabled, concatenates a quantum feature
               extracted from the flattened patch.

        Returns:
            Tensor of shape (batch, 1, H-k+1, W-k+1) if quantum disabled,
            otherwise shape (batch, 2, H-k+1, W-k+1) where the second channel
            contains the quantum feature.
        """
        conv_out = self.conv(x)  # shape (batch, 1, H-k+1, W-k+1)
        if not self.use_quantum:
            return conv_out

        # quantum feature extraction
        batch, _, h_out, w_out = conv_out.shape
        # Extract patches of size kernel_size from original input
        patches = torch.nn.functional.unfold(x, kernel_size=self.kernel_size,
                                             stride=1, padding=0)
        # patches shape: (batch, kernel_size*kernel_size, H-k+1)*(W-k+1)
        patches = patches.permute(0, 2, 1)  # (batch, N, k*k)
        N = patches.shape[1]
        quantum_features = []
        for i in range(batch):
            qfeat_batch = []
            for j in range(N):
                patch_np = patches[i, j].cpu().numpy()
                qfeat = self.quantum_extractor.run(patch_np)
                qfeat_batch.append(qfeat)
            qfeat_arr = torch.tensor(qfeat_batch, dtype=torch.float32,
                                     device=x.device)
            quantum_features.append(qfeat_arr.unsqueeze(0))
        quantum_features = torch.cat(quantum_features, dim=0)  # (batch, N)
        quantum_features = quantum_features.view(batch, h_out, w_out, 1).permute(0, 3, 1, 2)
        return torch.cat([conv_out, quantum_features], dim=1)

def Conv():
    """Drop‑in replacement that returns a ConvHybrid instance with
    classical behaviour only (use_quantum=False)."""
    return ConvHybrid(kernel_size=2, use_quantum=False)

__all__ = ["ConvHybrid", "Conv"]
