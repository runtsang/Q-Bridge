"""Hybrid quantum convolution module inspired by Conv.py and QuanvolutionFilter.

The class implements a parameterized quantum circuit that mirrors
the 2×2 kernel size of the classical counterpart.  Each pixel in a
patch is encoded as a rotation angle; values above a threshold
trigger a π rotation, otherwise 0.  The circuit is a shallow random
mixing layer followed by measurement in the Z basis.  The output is
the average probability of measuring |1> across all qubits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit


class HybridQuantumConv:
    """
    Quantum filter that operates on 2×2 image patches.

    Parameters
    ----------
    kernel_size : int
        Size of the image patch (2 for compatibility with the
        classical filter).  The number of qubits is kernel_size².
    backend : qiskit.providers.BaseBackend
        Execution backend.  Defaults to the Aer qasm simulator.
    shots : int
        Number of shots used to estimate measurement probabilities.
    threshold : float
        Pixel intensity threshold.  Values greater than this trigger
        a π rotation in the encoding.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.BaseBackend | None = None,
        shots: int = 200,
        threshold: float = 0.5,
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

        # Build the parameterized circuit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
        ]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        # Random mixing layer of depth 2
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single 2×2 patch.

        Parameters
        ----------
        data : np.ndarray
            2×2 array of pixel intensities (shape (2,2)).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        # Flatten the patch into a 1D array
        flat = data.reshape(1, self.n_qubits)

        # Bind parameters based on threshold
        param_binds = []
        for row in flat:
            bind = {
                self.theta[i]: np.pi if val > self.threshold else 0
                for i, val in enumerate(row)
            }
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        # Compute average probability of |1> over all qubits
        total_ones = 0
        for bitstring, count in result.items():
            ones = sum(int(bit) for bit in bitstring)
            total_ones += ones * count

        return total_ones / (self.shots * self.n_qubits)


__all__ = ["HybridQuantumConv"]
