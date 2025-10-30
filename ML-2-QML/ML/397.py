"""ConvEnhanced: hybrid classical‑quantum convolution filter.

This module extends the original Conv filter by adding a learnable bias
layer, an adaptive threshold, and the ability to operate in classic,
quantum, or hybrid mode. It can be used as a drop‑in replacement for
the original Conv class.

"""

from __future__ import annotations

import time
import torch
from torch import nn
import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter

__all__ = ["ConvEnhanced"]

class ConvEnhanced(nn.Module):
    """Hybrid classical‑quantum convolution filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, mode: str = "classic") -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.mode = mode
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=False)
        # Learnable per‑pixel bias
        self.bias = nn.Parameter(torch.zeros(kernel_size, kernel_size))
        # Adaptive threshold
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        # Hybrid weight
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # Quantum backend
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def _quantum_output(self, data: np.ndarray) -> float:
        """Run a simple quantum filter on the input data."""
        n_qubits = self.kernel_size ** 2
        qc = qiskit.QuantumCircuit(n_qubits)
        theta = [Parameter(f"θ{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        # Entangle with a simple CX chain
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        # Bind parameters based on threshold
        param_binds = []
        for val in data.flatten():
            bind = {theta[i]: np.pi if val > self.threshold.item() else 0.0 for i, val in enumerate(data.flatten())}
            param_binds.append(bind)
        job = execute(qc, backend=self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(qc)
        total = 0
        for bitstring, count in result.items():
            ones = bitstring.count("1")
            total += ones * count
        return total / (self.shots * n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the filter."""
        # Expect x shape: (batch, 1, H, W)
        conv_out = self.conv(x)
        bias_tensor = self.bias.view(1, 1, self.kernel_size, self.kernel_size)
        conv_out = conv_out + bias_tensor
        activated = torch.sigmoid(conv_out - self.threshold)
        mean_act = activated.mean(dim=[2, 3])  # batch-wise mean

        if self.mode == "classic":
            return mean_act
        elif self.mode == "quantum":
            # Use the first patch of each sample
            batch_size = x.size(0)
            q_outs = []
            for i in range(batch_size):
                patch = x[i, 0, :self.kernel_size, :self.kernel_size].detach().cpu().numpy()
                q_outs.append(self._quantum_output(patch))
            return torch.tensor(q_outs, device=x.device, dtype=torch.float32)
        elif self.mode == "hybrid":
            classic = mean_act
            # For hybrid, use quantum output on a single patch from the first sample
            patch = x[0, 0, :self.kernel_size, :self.kernel_size].detach().cpu().numpy()
            quantum = self._quantum_output(patch)
            hybrid = self.alpha * classic + (1 - self.alpha) * quantum
            return hybrid
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def benchmark(self, x: torch.Tensor, mode: str | None = None) -> float:
        """Run a quick benchmark of the selected mode and return elapsed time."""
        if mode:
            self.mode = mode
        start = time.time()
        self.forward(x)
        return time.time() - start
