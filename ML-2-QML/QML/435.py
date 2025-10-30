"""Quantum convolution generator that mirrors the classical ConvGen064 interface.

The class builds a variational circuit per kernel size and evaluates the
average probability of measuring |1> over all qubits.  It can be dropped in
place of the classical filter when a quantum backend is desired.

Author: gpt‑oss‑20b
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

__all__ = ["ConvGen064"]

class ConvGen064:
    """Hybrid quantum convolution filter.

    Parameters
    ----------
    kernel_sizes : list[int] | None
        Kernel sizes to support; defaults to [2].
    backend : qiskit.providers.BaseBackend | None
        Quantum backend to execute the circuit.  If None, the local Aer
        simulator is used.
    shots : int
        Number of shots per evaluation.
    threshold : float
        Threshold for binarizing input data when binding parameters.
    """

    def __init__(
        self,
        kernel_sizes: list[int] | None = None,
        backend=None,
        shots: int = 100,
        threshold: float = 127,
    ) -> None:
        self.kernel_sizes = kernel_sizes if kernel_sizes is not None else [2]
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        # Build a circuit for each kernel size
        self.circuits = {}
        for ks in self.kernel_sizes:
            self.circuits[ks] = self._build_circuit(ks)

    def _build_circuit(self, ks: int):
        n_qubits = ks ** 2
        qc = qiskit.QuantumCircuit(n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n_qubits, 2)
        qc.measure_all()
        return qc, theta

    def run(self, data) -> float:
        """Evaluate the quantum filter on a 2D array.

        Parameters
        ----------
        data : array-like
            2D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data = np.asarray(data, dtype=np.float32)
        ks = data.shape[0]
        if ks not in self.kernel_sizes:
            raise ValueError(f"Unsupported kernel size {ks}. Supported: {self.kernel_sizes}")

        qc, theta = self.circuits[ks]
        # Bind parameters based on threshold
        param_binds = []
        for val in data.flatten():
            bind = {theta[i]: np.pi if val > self.threshold else 0}
            param_binds.append(bind)

        job = qiskit.execute(
            qc,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(qc)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * ks ** 2)
