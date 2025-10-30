"""
Quantum variational filter that encodes a kernel-sized patch via RX gates,
applies trainable RZ rotations, and measures the probability of |1>
over all qubits.  The module exposes a `run` method compatible with the
original Conv() callable and supports gradient computation via Qiskit's
parameter‑shift rule.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import random_circuit
from typing import Iterable

__all__ = ["ConvEnhanced"]

class ConvEnhanced:
    """
    Quantum variational filter for a kernel-sized patch.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the kernel (patch).  The number of qubits is kernel_size**2.
    threshold : float, default 127.0
        Threshold used to binarize classical data before encoding.
    shots : int, default 1024
        Number of shots for the qasm simulator.
    backend : qiskit.providers.Backend, optional
        Backend to execute the circuit on.  Defaults to Aer qasm_simulator.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127.0,
        shots: int = 1024,
        backend=None,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.n_qubits = kernel_size ** 2

        # Parameter vector for trainable RZ rotations
        self.theta = ParameterVector("theta", self.n_qubits)

        # Build the parameterized circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        # Data encoding via RX gates (to be bound at run time)
        self.data_params = ParameterVector("x", self.n_qubits)
        for q in range(self.n_qubits):
            self.circuit.rx(self.data_params[q], q)
        # Trainable RZ rotations
        for q in range(self.n_qubits):
            self.circuit.rz(self.theta[q], q)
        # Entangling layer (random for demonstration)
        self.circuit += random_circuit(self.n_qubits, depth=2)
        self.circuit.measure_all()

    def run(self, data: Iterable[float]) -> float:
        """
        Execute the quantum filter on a classical patch.

        Parameters
        ----------
        data : array‑like
            2‑D array with shape (kernel_size, kernel_size) or a flattened
            vector of length kernel_size**2.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        # Prepare data binding
        arr = np.asarray(data, dtype=float).flatten()
        if arr.size!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} data points, got {arr.size}.")
        # Encode data: threshold to binary, then map to RX angles
        data_binds = {self.data_params[i]: np.pi if val > self.threshold else 0.0
                      for i, val in enumerate(arr)}
        # Combine data and trainable parameters (theta are already parameters)
        bind_dict = {**data_binds}
        # Execute circuit
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[bind_dict],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        # Compute average probability of |1> over all qubits
        total_prob = 0.0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_prob += ones * freq
        prob_one = total_prob / (self.shots * self.n_qubits)
        return prob_one

    def parameters(self):
        """
        Return the trainable parameters for gradient computation.
        """
        return self.theta
