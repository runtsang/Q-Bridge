"""ConvGen258: quantum convolution filter using Qiskit."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

__all__ = ["ConvGen258"]


class ConvGen258:
    """
    Quantum convolution filter that emulates a classical 2‑D filter by
    mapping each input pixel to a qubit rotation and measuring the
    probability of observing |1>.  The design supports multiple kernel
    sizes and configurable circuit depth.

    Parameters
    ----------
    kernel_sizes : list[int] or tuple[int,...], optional
        Kernel sizes to support.  Defaults to ``[2]``.
    threshold : float, default 0.5
        Threshold used to decide the rotation angle (π if above, 0 otherwise).
    shots : int, default 1024
        Number of shots for the quantum execution.
    depth : int, default 2
        Depth of the random circuit appended after the RX rotations.
    backend_name : str, default "qasm_simulator"
        Qiskit backend name.
    """

    def __init__(
        self,
        kernel_sizes: list[int] | tuple[int,...] | None = None,
        threshold: float = 0.5,
        shots: int = 1024,
        depth: int = 2,
        backend_name: str = "qasm_simulator",
    ) -> None:
        self.kernel_sizes = kernel_sizes or [2]
        self.threshold = threshold
        self.shots = shots
        self.depth = depth

        # Load the backend
        self.backend = qiskit.Aer.get_backend(backend_name)

        # Pre‑build a circuit for each kernel size
        self.circuits = {}
        for k in self.kernel_sizes:
            self.circuits[k] = self._build_circuit(k)

    def _build_circuit(self, k: int) -> qiskit.QuantumCircuit:
        """Create a parameterized quantum circuit for a k×k kernel."""
        n = k * k
        circ = qiskit.QuantumCircuit(n)
        thetas = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(n)]
        for i in range(n):
            circ.rx(thetas[i], i)
        circ.barrier()
        for _ in range(self.depth):
            circ += random_circuit(n, 2)
        circ.measure_all()
        return circ

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum filter on a 2‑D array.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (k, k) where k is one of the supported kernel sizes.

        Returns
        -------
        float
            Mean probability of measuring |1> across all qubits and shots.
        """
        data = np.asarray(data)
        if data.ndim!= 2:
            raise ValueError("Input must be a 2‑D array")

        k = data.shape[0]
        if k not in self.circuits:
            raise ValueError(f"Kernel size {k} not supported")

        circ = self.circuits[k]
        # Bind parameters based on threshold
        param_bind = {
            circ.parameters[i]: np.pi if val > self.threshold else 0.0
            for i, val in enumerate(data.flatten())
        }

        job = qiskit.execute(
            circ,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(circ)

        total_ones = 0
        total_counts = 0
        for bitstring, c in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * c
            total_counts += c

        mean_prob = total_ones / (total_counts * k)
        return mean_prob
