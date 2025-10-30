"""ConvEnhanced: hybrid classical/quantum convolutional filter.

This module extends the original Conv seed by providing a full
nn.Module that can be inserted into a PyTorch model.  The filter can
operate either with a standard 2‑D convolution or with a Qiskit
variational circuit.  The class exposes a `forward` method that
automatically dispatches to the chosen backend and a `loss_fn`
that adds a small noise‑aware penalty when the quantum backend is
active.  The implementation supports multi‑channel inputs by using
grouped convolutions and by running the quantum filter independently
for each channel.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector


# --------------------------------------------------------------------------- #
#  Classical convolutional filter
# --------------------------------------------------------------------------- #
class _ClassicalConv(nn.Module):
    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # grouped convolution to preserve each channel independently
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            bias=True,
            groups=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution followed by a sigmoid threshold."""
        logits = self.conv(x)
        return torch.sigmoid(logits - self.threshold)


# --------------------------------------------------------------------------- #
#  Quantum convolutional filter
# --------------------------------------------------------------------------- #
class _QuantumConv:
    """Quantum filter that operates on a 2‑D kernel of size k×k."""

    def __init__(
        self,
        kernel_size: int,
        threshold: float = 0.0,
        shots: int = 1024,
        backend: Optional[object] = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._build_circuit()

    def _build_circuit(self) -> None:
        self.theta = ParameterVector("theta", self.n_qubits)
        self.circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        # add a shallow random entangling layer
        self.circuit += qiskit.circuit.random.random_circuit(
            self.n_qubits, 2, measure=False
        )
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Run the circuit on a single kernel.

        Args:
            data: 1‑D array of length n_qubits with values in [0, 255].

        Returns:
            float: average probability of measuring |1> across all qubits.
        """
        bind = {}
        for i, val in enumerate(data):
            bind[self.theta[i]] = math.pi if val > self.threshold else 0.0
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        # compute average probability of |1>
        total = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total += ones * freq
        return total / (self.shots * self.n_qubits)


# --------------------------------------------------------------------------- #
#  Hybrid convolutional filter
# --------------------------------------------------------------------------- #
class ConvEnhanced(nn.Module):
    """
    Hybrid classical/quantum convolutional filter.

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel (k×k).
    in_channels : int
        Number of input channels.  The filter is applied independently
        to each channel.
    threshold : float
        Threshold used for both classical sigmoid and quantum
        angle encoding.
    use_quantum : bool
        If True the filter uses the quantum backend; otherwise it
        falls back to the classical convolution.
    shots : int
        Number of shots for the quantum simulator.
    backend : qiskit.providers.Backend, optional
        Quantum backend.  If None, the Aer qasm simulator is used.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        threshold: float = 0.0,
        use_quantum: bool = False,
        shots: int = 1024,
        backend: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.threshold = threshold
        self.use_quantum = use_quantum
        self.shots = shots
        self.backend = backend

        # classical sub‑module
        self.classical = _ClassicalConv(kernel_size, in_channels, threshold)

        # quantum sub‑module
        self.quantum = _QuantumConv(
            kernel_size, threshold, shots, backend
        )

    # --------------------------------------------------------------------- #
    #  Forward passes
    # --------------------------------------------------------------------- #
    def forward_classical(self, x: torch.Tensor) -> torch.Tensor:
        """Pure classical forward."""
        return self.classical(x)

    def forward_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """Quantum forward that operates on each kernel independently."""
        batch, channels, H, W = x.shape
        # unfold the input to obtain all k×k patches
        patches = F.unfold(x, kernel_size=self.kernel_size)
        L = patches.shape[-1]  # number of patches
        patches = patches.reshape(batch, channels, self.kernel_size**2, L)
        outputs = torch.empty(batch, channels, L, dtype=torch.float32)

        for b in range(batch):
            for c in range(channels):
                data = patches[b, c].transpose(0, 1).detach().cpu().numpy()
                probs = []
                for p in range(L):
                    probs.append(self.quantum.run(data[:, p]))
                outputs[b, c] = torch.tensor(probs)

        out_H = H - self.kernel_size + 1
        out_W = W - self.kernel_size + 1
        return outputs.reshape(batch, channels, out_H, out_W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch to the chosen backend."""
        if self.use_quantum:
            return self.forward_quantum(x)
        return self.forward_classical(x)

    # --------------------------------------------------------------------- #
    #  Training utilities
    # --------------------------------------------------------------------- #
    def loss_fn(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean‑squared error with a small noise penalty for the quantum case."""
        mse = F.mse_loss(outputs, targets)
        if self.use_quantum:
            # a toy noise penalty: encourage output magnitudes to stay small
            penalty = 0.01 * torch.mean(outputs.detach() ** 2)
            return mse + penalty
        return mse

    def set_use_quantum(self, flag: bool) -> None:
        """Switch the backend at runtime."""
        self.use_quantum = flag

    def __repr__(self) -> str:
        backend = (
            self.backend.__class__.__name__ if self.backend else "Aer qasm_simulator"
        )
        return (
            f"{self.__class__.__name__}(kernel_size={self.kernel_size}, "
            f"in_channels={self.in_channels}, threshold={self.threshold}, "
            f"use_quantum={self.use_quantum}, shots={self.shots}, "
            f"backend={backend})"
        )


__all__ = ["ConvEnhanced"]
