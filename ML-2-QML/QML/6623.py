"""ConvHybrid: quantum convolutional filter.

This module implements a quantum version of the ConvHybrid filter.
It builds a parameter‑free circuit that performs Rx rotations on each
qubit, applies a short random circuit, measures all qubits, and
returns the average probability of measuring |1> across the qubits.
The circuit is executed on a chosen backend (default Aer simulator)
and can be used for non‑linear activation in hybrid models.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit.random import random_circuit

__all__ = ["ConvHybrid"]


class ConvHybrid:
    """Quantum implementation of the ConvHybrid filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel (number of qubits = kernel_size**2).
    backend : qiskit.providers.Backend, optional
        Quantum backend to execute the circuit. Defaults to Aer
        simulator if available.
    shots : int, default 100
        Number of shots per execution.
    threshold : float, default 127
        Threshold used to decide whether to rotate a qubit by π.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 100,
        threshold: float = 127,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """Execute the quantum circuit on the provided data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with integer pixel
            values in the range 0–255.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {
                self._circuit.parameters[i]: np.pi if val > self.threshold else 0
                for i, val in enumerate(dat)
            }
            param_binds.append(bind)

        job = qiskit.execute(
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
