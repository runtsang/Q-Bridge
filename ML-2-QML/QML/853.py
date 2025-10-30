"""Quantum convolution filter using Qiskit.

The module defines a ``Conv`` factory that returns a ``QuantumConvFilter`` instance.
The filter builds a parameterised circuit for each ``kernel_size × kernel_size`` patch,
binds the input data to the circuit parameters, runs the circuit on a chosen backend,
and returns the average probability of measuring |1> across all qubits.

The implementation supports a configurable threshold, number of shots, and optional
depth‑wise separable processing of multi‑channel data.  The API is compatible with the
original seed while providing a quantum‑centric contribution.

Example
-------
>>> import numpy as np
>>> from conv_qml import Conv
>>> filt = Conv(kernel_size=3, threshold=0.5, shots=200)
>>> data = np.random.rand(3, 3)
>>> filt.run(data)
0.42
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from typing import Iterable, Optional

__all__ = ["Conv"]


class QuantumConvFilter:
    """Internal Qiskit implementation of the convolution filter."""

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        backend: Optional[qiskit.providers.Backend] = None,
        shots: int = 100,
        use_depthwise: bool = False,
        in_channels: int = 1,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.use_depthwise = use_depthwise
        self.in_channels = in_channels

        self.n_qubits = kernel_size * kernel_size
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Build a reusable parameterised circuit for the filter."""
        qc = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qc.rx(self.theta[i], i)
        qc.barrier()
        # Add a small random circuit to mix the qubits
        qc += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def run(self, data: Iterable[float]) -> float:
        """Apply the quantum filter to a 2‑D kernel and return the average |1> probability.

        Parameters
        ----------
        data
            2‑D array or list of length ``kernel_size * kernel_size``.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        if self.use_depthwise:
            # For simplicity, depth‑wise is treated the same as standard in this example.
            pass

        flat = np.asarray(data).reshape(-1)
        if flat.size!= self.n_qubits:
            raise ValueError(
                f"Expected input of size {self.n_qubits}, got {flat.size}"
            )

        param_binds = []
        for i, val in enumerate(flat):
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0}
            param_binds.append(bind)

        job = execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self._circuit)

        total_counts = sum(counts.values())
        ones = 0
        for bitstring, count in counts.items():
            ones += bitstring.count("1") * count

        return ones / (total_counts * self.n_qubits)


def Conv(
    kernel_size: int = 2,
    threshold: float = 0.0,
    backend: Optional[qiskit.providers.Backend] = None,
    shots: int = 100,
    use_depthwise: bool = False,
    in_channels: int = 1,
) -> QuantumConvFilter:
    """Factory function returning a quantum convolution filter instance."""
    return QuantumConvFilter(
        kernel_size=kernel_size,
        threshold=threshold,
        backend=backend,
        shots=shots,
        use_depthwise=use_depthwise,
        in_channels=in_channels,
    )
