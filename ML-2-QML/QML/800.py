"""Quantum kernel using a variational ansatz and Qiskit state‑vector simulator."""

from __future__ import annotations

import numpy as np
import torch
from typing import Sequence

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

class KernalAnsatz:
    """Variational ansatz that encodes two classical vectors."""
    def __init__(self, n_qubits: int = 4, layers: int = 1) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.params = [Parameter(f"θ_{i}") for i in range(n_qubits * layers)]
        self.circuit = QuantumCircuit(n_qubits)
        for l in range(layers):
            for q in range(n_qubits):
                self.circuit.ry(self.params[l * n_qubits + q], q)
            for q in range(n_qubits - 1):
                self.circuit.cx(q, q + 1)

    def encode(self,
               circuit: QuantumCircuit,
               vec: torch.Tensor,
               sign: int = 1) -> None:
        """Apply a data‑encoding layer."""
        for q, val in enumerate(vec):
            circuit.ry(sign * float(val), q)

    def __call__(self,
                 x: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value |⟨ψ(x)|ψ(y)⟩|²."""
        qc_x = QuantumCircuit(self.n_qubits)
        qc_y = QuantumCircuit(self.n_qubits)
        self.encode(qc_x, x, sign=1)
        self.encode(qc_y, y, sign=-1)
        qc_x.append(self.circuit, range(self.n_qubits))
        qc_y.append(self.circuit, range(self.n_qubits))
        backend = Aer.get_backend("statevector_simulator")
        sv_x = execute(qc_x, backend).result().get_statevector()
        sv_y = execute(qc_y, backend).result().get_statevector()
        overlap = np.vdot(sv_x, sv_y)
        return torch.tensor(abs(overlap)**2, dtype=torch.float32)

class Kernel:
    """Wrapper that exposes a mini‑batch Gram‑matrix builder."""
    def __init__(self, n_qubits: int = 4, layers: int = 1) -> None:
        self.ansatz = KernalAnsatz(n_qubits, layers)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

    def gram_matrix(self,
                    X: torch.Tensor,
                    Y: torch.Tensor,
                    batch_size: int = 32) -> torch.Tensor:
        """Compute the Gram matrix using the quantum kernel."""
        n, _ = X.shape
        m, _ = Y.shape
        K = torch.empty((n, m), dtype=torch.float32)
        for i in range(0, n, batch_size):
            x_batch = X[i:i+batch_size]
            for j in range(0, m, batch_size):
                y_batch = Y[j:j+batch_size]
                K[i:i+batch_size, j:j+batch_size] = self(x_batch, y_batch)
        return K

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  n_qubits: int = 4,
                  layers: int = 1,
                  batch_size: int = 32) -> np.ndarray:
    """Return the Gram matrix as a NumPy array."""
    kernel = Kernel(n_qubits, layers)
    X = torch.stack([x.squeeze() for x in a])
    Y = torch.stack([y.squeeze() for y in b])
    return kernel.gram_matrix(X, Y).cpu().numpy()

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
