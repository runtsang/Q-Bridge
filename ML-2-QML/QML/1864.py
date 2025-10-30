"""Variational quantum convolutional filter.

The quantum implementation mirrors the classical Conv class and can be used inside a
torch.nn.Sequential pipeline.  It supports:
- Arbitrary kernel size (up to 10 qubits in practice).
- A simple RY+CX variational ansatz.
- Optional noise model and real‑backend execution.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter
from typing import Any, Optional

class Conv:
    """Quantum convolution using a variational circuit."""
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        backend: Optional[Any] = None,
        shots: int = 1024,
        noise_model: Optional[Any] = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size * kernel_size
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.noise_model = noise_model

        # Build a simple variational circuit
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits)

        # Parameterised rotations
        for i in range(self.n_qubits):
            self.circuit.ry(self.theta[i], i)

        # Entangling layer (ring of CX)
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.cx(self.n_qubits - 1, 0)

        self.circuit.barrier()
        self.circuit.measure_all()

    def forward(self, patch: np.ndarray) -> float:
        """Run the circuit on a kernel patch.

        Args:
            patch: 2D numpy array of shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across all qubits.
        """
        if patch.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError(
                f"Expected patch shape {(self.kernel_size, self.kernel_size)}, "
                f"got {patch.shape}"
            )

        flat = patch.reshape((1, self.n_qubits))

        param_binds = []
        for dat in flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
            noise_model=self.noise_model,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        total = 0
        for bitstring, count in counts.items():
            ones = bitstring.count("1")
            total += ones * count

        prob = total / (self.shots * self.n_qubits)
        return prob

    def __repr__(self) -> str:
        return (
            f"Conv(kernel_size={self.kernel_size}, "
            f"threshold={self.threshold}, "
            f"backend={self.backend.__class__.__name__})"
        )

__all__ = ["Conv"]
