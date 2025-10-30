"""HybridLayer combining quantum convolution and fully connected circuits.

The quantum implementation builds a composite circuit that first
performs a quanvolution (parameterized Rx gates + random entanglement)
followed by a single‑qubit Ry for the fully connected step.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from typing import Iterable


class HybridLayer:
    """Quantum hybrid layer: QuanvCircuit → Ry(theta) → Measure."""

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 100,
        threshold: float = 127,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2 + 1  # +1 for the fully connected qubit
        self.shots = shots
        self.threshold = threshold

        # Build the circuit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        # Parameters for the quanvolution part
        self.quanv_params = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(kernel_size ** 2)
        ]
        for i in range(kernel_size ** 2):
            self._circuit.rx(self.quanv_params[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(kernel_size ** 2, 2)
        # Fully connected parameter on the last qubit
        self.fc_param = qiskit.circuit.Parameter("fc_theta")
        self._circuit.ry(self.fc_param, self.n_qubits - 1)
        self._circuit.barrier()
        self._circuit.measure_all()

        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

    def run(self, thetas: Iterable[float], data: np.ndarray) -> np.ndarray:
        """
        Execute the hybrid quantum circuit.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameters for the fully connected Ry gate (length 1).
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with values in [0, 255].

        Returns
        -------
        np.ndarray
            Array containing the convolution expectation (probability of |1>
            across the convolution qubits) followed by the fully connected
            expectation (probability of |1> on the last qubit).
        """
        # Prepare parameter binds
        data_flat = data.reshape(-1)
        param_binds = []
        for val in data_flat:
            bind = {}
            for i, param in enumerate(self.quanv_params):
                bind[param] = np.pi if val > self.threshold else 0
            bind[self.fc_param] = thetas[0]
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        # Compute expectations
        conv_counts = 0
        fc_counts = 0
        total_counts = 0
        for key, val in result.items():
            bits = [int(b) for b in key]
            conv_bits = bits[: self.kernel_size ** 2]
            fc_bit = bits[-1]
            conv_counts += sum(conv_bits) * val
            fc_counts += fc_bit * val
            total_counts += val

        conv_expectation = conv_counts / (total_counts * self.kernel_size ** 2)
        fc_expectation = fc_counts / total_counts

        return np.array([conv_expectation, fc_expectation])


__all__ = ["HybridLayer"]
