"""
HybridNet – Quantum implementation of a convolution + fully‑connected circuit.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import execute


class HybridQuantumCircuit:
    """
    Quantum circuit that mirrors the hybrid classical architecture:
    - A “quanvolution” block of RX gates parameterised per qubit.
    - A random entangling layer to introduce non‑linear correlations.
    - A fully‑connected block of RY gates that act as a linear layer.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 backend=None,
                 shots: int = 100,
                 threshold: float = 127.0) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        # Parameters for the quanvolution (RX) block
        self.theta_conv = [qiskit.circuit.Parameter(f"theta_conv{i}") for i in range(self.n_qubits)]
        # Parameters for the fully‑connected (RY) block
        self.theta_fc   = [qiskit.circuit.Parameter(f"theta_fc{i}")   for i in range(self.n_qubits)]

        self._circuit = qiskit.QuantumCircuit(self.n_qubits)

        # Quanvolution block
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta_conv[i], i)

        self._circuit.barrier()

        # Entangling layer
        self._circuit += random_circuit(self.n_qubits, depth=2)

        # Fully‑connected block
        for i in range(self.n_qubits):
            self._circuit.ry(self.theta_fc[i], i)

        self._circuit.measure_all()

    def run(self,
            data: np.ndarray,
            conv_thetas: Iterable[float],
            fc_thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the hybrid circuit on a single data patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size); flattened to
            a vector of length n_qubits.
        conv_thetas : Iterable[float]
            Parameters for the RX (quanvolution) block.
        fc_thetas : Iterable[float]
            Parameters for the RY (fully‑connected) block.

        Returns
        -------
        np.ndarray
            Scalar expectation value representing the average probability of
            measuring |1> across all qubits.
        """
        # Flatten data to match qubit ordering
        flat = np.reshape(data, (1, self.n_qubits))

        # Build parameter bindings for each sample
        param_binds = []
        for dat in flat:
            bind = {}
            for i, val in enumerate(dat):
                # Use data to set a threshold‑based angle if desired; otherwise use conv_thetas directly
                bind[self.theta_conv[i]] = conv_thetas[i]
                bind[self.theta_fc[i]]   = fc_thetas[i]
            param_binds.append(bind)

        job = execute(self._circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        # Compute expectation as average |1> probability
        total_ones = 0
        for key, count in result.items():
            ones = sum(int(bit) for bit in key)
            total_ones += ones * count

        expectation = total_ones / (self.shots * self.n_qubits)
        return np.array([expectation])


__all__ = ["HybridQuantumCircuit"]
