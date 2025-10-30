"""Quantum patch filter module.

Provides `QuantumPatchFilter`, a thin wrapper around a Qiskit circuit that
encodes a 2×2 image patch into a small register of qubits, applies a
random circuit of configurable depth, and measures all qubits.  The
output is the average probability of measuring |1> across all qubits
and shots, which is treated as a classical feature value.

This implementation mirrors the quantum seed in the repository
and extends it by allowing the depth of random layers to be tuned
and by exposing a convenient `run` method that accepts a NumPy
array representing a single patch.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit.providers.aer import AerSimulator


class QuantumPatchFilter:
    """Quantum filter for a single image patch.

    Parameters
    ----------
    kernel_size : int
        Size of the square patch (default 2).
    backend : str
        Qiskit simulator backend name (default 'qasm_simulator').
    shots : int
        Number of shots for measurement (default 100).
    threshold : float
        Threshold for deciding whether to apply a π rotation
        to a qubit based on pixel intensity (default 127).
    depth : int
        Number of random layers added to the circuit (default 2).
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: str = "qasm_simulator",
        shots: int = 100,
        threshold: float = 127,
        depth: int = 2,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size**2
        self.backend = AerSimulator(name=backend)
        self.shots = shots
        self.threshold = threshold
        self.depth = depth

        # Build the parameterised circuit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
        ]

        # Encode each pixel with an RX gate
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)

        self._circuit.barrier()

        # Add random layers
        self._circuit += random_circuit(self.n_qubits, depth=self.depth)

        self._circuit.measure_all()

    def run(self, patch: np.ndarray) -> float:
        """Run the quantum circuit on a single patch.

        Parameters
        ----------
        patch : np.ndarray
            2D array of shape ``(kernel_size, kernel_size)`` with pixel
            intensities in the range [0, 255].

        Returns
        -------
        float
            Average probability of measuring |1> over all qubits and shots.
        """
        flat = patch.reshape(-1)
        param_bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(flat)}

        job = qiskit.execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self._circuit)

        total_ones = 0
        for bitstring, count in counts.items():
            ones = sum(int(b) for b in bitstring)
            total_ones += ones * count

        return total_ones / (self.shots * self.n_qubits)

    def __repr__(self) -> str:
        return (
            f"QuantumPatchFilter(kernel_size={self.kernel_size}, "
            f"shots={self.shots}, depth={self.depth})"
        )


__all__ = ["QuantumPatchFilter"]
