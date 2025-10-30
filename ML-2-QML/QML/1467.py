"""ConvHybrid: quantum filter with variational ansatz."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter
from typing import Optional

class ConvHybrid:
    """
    Quantum convolutional filter employing a variational circuit.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square patch to encode.
    backend : qiskit.providers.Backend, optional
        Backend to execute the circuit; defaults to AerSimulator.
    shots : int, default 100
        Number of shots for sampling.
    threshold : float, default 127
        Threshold for mapping pixel values to rotation angles.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: Optional[qiskit.providers.Backend] = None,
        shots: int = 100,
        threshold: float = 127,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or AerSimulator()
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        """
        Construct a parameterised circuit that encodes the input data
        into rotation angles and applies a shallow entangling ansatz.
        """
        qc = qiskit.QuantumCircuit(self.n_qubits)
        # Data‑dependent rotations
        theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qc.ry(theta[i], i)
        qc.barrier()
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.barrier()
        # Second layer of rotations (trainable parameters)
        phi = [Parameter(f"phi_{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qc.rz(phi[i], i)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single data patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> over all qubits and shots.
        """
        flat = data.reshape(1, self.n_qubits)

        param_binds = []
        for sample in flat:
            bind = {}
            for i, val in enumerate(sample):
                angle = np.pi if val > self.threshold else 0.0
                bind[self._circuit.parameters[i]] = angle
            # For the second layer parameters we leave them at zero (untrained)
            for i in range(self.n_qubits, 2 * self.n_qubits):
                bind[self._circuit.parameters[i]] = 0.0
            param_binds.append(bind)

        transpiled = transpile(self._circuit, self.backend)
        qobj = assemble(transpiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts(transpiled)

        total_ones = 0
        total_counts = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq
            total_counts += freq

        return total_ones / (total_counts * self.n_qubits)

def Conv() -> ConvHybrid:
    """
    Factory function matching the original API.

    Returns
    -------
    ConvHybrid
        Instance of the quantum filter.
    """
    return ConvHybrid(kernel_size=2, shots=200, threshold=127)

__all__ = ["ConvHybrid", "Conv"]
