"""Quantum convolution module that implements a parameterised variational filter.

The class `ConvFilter` emulates the interface of the classical filter but
uses a small parameterised circuit that can be executed on any Qiskit
backend.  The circuit is constructed from RX rotations that encode the input
data and a fixed two‑layer random circuit.  The output is the average
probability of measuring |1⟩ across all qubits."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit.random import random_circuit


class ConvFilter:
    """
    Quantum implementation of a 2‑D convolution filter.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.5,
        shots: int = 200,
        backend=None,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square filter (determines number of qubits).
        threshold : float
            Data threshold used to map pixel values to rotation angles.
        shots : int
            Number of measurement shots to run.
        backend : qiskit.providers.basebackend.BaseBackend | None
            Backend to execute the circuit on.  Defaults to the Aer qasm simulator.
        """
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Build the parameterised circuit once during initialisation.
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Create the core variational circuit used for every forward pass."""
        n_qubits = self.kernel_size ** 2
        qc = QuantumCircuit(n_qubits, n_qubits)

        # Parameters that encode the input data.
        self.theta = [Parameter(f"theta{i}") for i in range(n_qubits)]

        # Data‑dependent RX rotations.
        for i in range(n_qubits):
            qc.rx(self.theta[i], i)

        # Add a small two‑layer random circuit for entanglement.
        qc = qc.compose(random_circuit(n_qubits, 2, seed=42))

        # Measure all qubits.
        qc.measure(range(n_qubits), range(n_qubits))

        self._circuit = qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum filter on an input array and return the average
        probability of measuring |1⟩ across all qubits.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size) containing
            pixel intensity values in the range [0, 255] for the original
            reference implementation.

        Returns
        -------
        float
            Mean probability of measuring |1⟩ across all qubits.
        """
        # Convert input to a flattened 1‑D array of length n_qubits.
        flat = np.reshape(data, (self.kernel_size ** 2,))

        # Map pixel values to rotation angles using the threshold.
        param_bindings = []
        for val in flat:
            angle = np.pi if val > self.threshold else 0.0
            param_bindings.append({theta: angle for theta in self.theta})

        # Execute the circuit for each data sample.
        job = execute(
            self._circuit,
            self.backend,
            parameter_binds=param_bindings,
            shots=self.shots,
            memory=False,
        )
        result = job.result()

        # Compute the average probability of measuring |1⟩.
        total_ones = 0
        for key, count in result.get_counts(self._circuit).items():
            # Key is a bitstring with qubit 0 as the leftmost bit.
            ones = key.count("1")
            total_ones += ones * count

        # Normalise by number of shots and qubits.
        return total_ones / (self.shots * self.kernel_size ** 2)
