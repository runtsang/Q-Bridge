"""ConvGen: variational quantum convolutional filter.

This module implements a parameter‑tuned variational circuit that replaces
the random circuit of the original QML seed. The circuit applies a
parameterized Ry rotation per qubit, followed by a simple entangling
layer, and measures all qubits. The run method maps input data to
parameter bindings and returns the average probability of measuring |1>
across all qubits, producing a probability‑like score.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from typing import Iterable, List, Tuple


class ConvGen:
    """
    Variational quantum convolutional filter with a learnable ansatz.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 1024,
        threshold: float = 127,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the convolutional kernel (kernel_size x kernel_size).
        backend : qiskit.providers.Backend, optional
            Qiskit backend to execute the circuit. Defaults to Aer qasm_simulator.
        shots : int
            Number of shots per execution.
        threshold : float
            Threshold used to map input pixel values to rotation angles.
        """
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Parameterized Ry rotations
        self.params: List[Parameter] = [
            Parameter(f"theta_{i}") for i in range(self.n_qubits)
        ]
        for i, p in enumerate(self.params):
            qc.ry(p, i)
        # Simple entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, data: Iterable[Iterable[float]]) -> float:
        """
        Execute the variational circuit on classical data.

        Parameters
        ----------
        data : 2‑D iterable of shape (kernel_size, kernel_size)
            Input pixel values.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data_arr = np.array(data, dtype=np.float32).reshape((self.n_qubits,))
        param_bindings = []
        for val in data_arr:
            bind = {}
            for i, p in enumerate(self.params):
                bind[p] = np.pi if val > self.threshold else 0.0
            param_bindings.append(bind)

        job = execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_bindings,
        )
        result = job.result().get_counts(self._circuit)

        total_ones = 0
        for bitstring, count in result.items():
            ones = sum(int(bit) for bit in bitstring)
            total_ones += ones * count

        return total_ones / (self.shots * self.n_qubits)

    def get_circuit(self) -> QuantumCircuit:
        """
        Return the underlying quantum circuit.
        """
        return self._circuit


# Public API
__all__ = ["ConvGen"]
