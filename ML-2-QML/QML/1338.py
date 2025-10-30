"""Quantum convolution filter for image patches.

The ConvFilter class implements a variational circuit that operates on
each pixel of a 2‑D patch.  The circuit uses RX rotations parameterized
by the pixel value (thresholded) followed by a layer of CZ gates.  The
output is the average probability of measuring |1> over all qubits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter, QuantumCircuit

class ConvFilter:
    """Hybrid convolution‑quantum filter (quantum variant)."""

    def __init__(
        self,
        kernel_size: int = 2,
        backend_name: str = "qasm_simulator",
        shots: int = 200,
        threshold: float = 0.0,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.backend = Aer.get_backend(backend_name)
        self.shots = shots
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Build a simple variational ansatz."""
        qc = QuantumCircuit(self.n_qubits)
        theta = [Parameter(f"θ{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        # entanglement layer
        for i in range(0, self.n_qubits - 1, 2):
            qc.cz(i, i + 1)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum circuit on classical data.

        Parameters
        ----------
        data : np.ndarray
            2D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across qubits.
        """
        flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for row in flat:
            bind = {}
            for i, val in enumerate(row):
                bind[self._circuit.parameters[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

def Conv():
    """Return a callable object that emulates the quantum filter."""
    return ConvFilter()

__all__ = ["Conv", "ConvFilter"]
