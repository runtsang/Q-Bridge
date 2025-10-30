"""Hybrid quantum convolutional filter that mirrors the classical interface.

The original `Conv` seed defined a quantum filter that measured the
average probability of |1> after a random circuit.  This extension
provides a reusable variational circuit template that can be
parameterised by the input data.  The class `ConvHybrid` can be
instantiated as a drop‑in replacement in a quantum neural network
and exposes the same `run` method as its classical counterpart.

The circuit uses a simple variational ansatz (ry rotations followed
by a few layers of CNOT entanglement).  Parameters are set to
either 0 or π depending on whether the corresponding input pixel
exceeds a threshold.  The output is the mean probability of
measuring |1> across all qubits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import Aer, execute

__all__ = ["ConvHybrid"]

class ConvHybrid:
    """Quantum hybrid convolutional filter.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the square kernel.  Defaults to 2.
    threshold : float, optional
        Threshold used to map pixel values to circuit parameters.
        Defaults to 127.
    shots : int, optional
        Number of shots for the simulator.  Defaults to 1024.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127.0,
        shots: int = 1024,
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Build a variational circuit template
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.ry(self.theta[i], i)
        # Simple entanglement pattern
        for i in range(0, self.n_qubits - 1, 2):
            self.circuit.cx(i, i + 1)
            self.circuit.cx(i + 1, i)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Execute the quantum circuit for the given 2‑D input.

        Parameters
        ----------
        data : np.ndarray
            Input array of shape ``(kernel_size, kernel_size)``.
        Returns
        -------
        float
            Mean probability of measuring |1> across all qubits.
        """
        # Flatten data to match qubit ordering
        flat = data.flatten()
        bind = {}
        for i, val in enumerate(flat):
            bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
        param_binds = [bind]

        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        total_ones = 0
        total_counts = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq
            total_counts += freq

        return total_ones / (total_counts * self.n_qubits)
