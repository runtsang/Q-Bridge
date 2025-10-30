"""Quantum counterpart of HybridFCL using Qiskit and TorchQuantum.

The quantum implementation contains a parameterised single‑qubit circuit
mirroring the original FCL example, an optional TorchQuantum encoder for
classical data, and a measurement that returns a single expectation value.
"""
from __future__ import annotations

import numpy as np
import qiskit
import torchquantum as tq
from torchquantum.functional import func_name_dict

# Optional helpers from the seed repo
from QuantumKernelMethod import Kernel as QKernel
from Conv import Conv  # reused for data preprocessing


class HybridFCLQuantum:
    """
    Quantum hybrid layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the core circuit.
    shots : int
        Number of shots for the simulator.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 100) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Optional classical preprocessing
        self.conv = Conv()
        self.kernel = QKernel()

    def run(self, thetas: list[float]) -> np.ndarray:
        """
        Execute the parameterised circuit and return an expectation value.

        Parameters
        ----------
        thetas : list[float]
            Parameters to bind to the circuit.

        Returns
        -------
        np.ndarray
            Expectation value as a one‑element array.
        """
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

    def kernel_matrix(
        self, a: list[torch.Tensor], b: list[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute a quantum kernel Gram matrix using the TorchQuantum encoder.
        """
        return np.array(
            [[self.kernel(x, y).item() for y in b] for x in a]
        )


__all__ = ["HybridFCLQuantum"]
