"""QML implementation of a parameterised fully connected layer using a variational circuit."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import Pauli


class ParametricFCL:
    """
    Parameterised quantum circuit that emulates a fully connected layer.

    Features
    --------
    * Multi‑qubit entanglement via a chain of CNOTs.
    * Depth‑controlled number of rotation layers.
    * Support for arbitrary backends and shot budgets.
    * Returns the expectation value of the Pauli‑Z operator on the first qubit.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 1024,
        depth: int = 2,
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.depth = depth

        # Create a parameter vector: one Ry per qubit per depth layer
        self.theta = ParameterVector("theta", length=n_qubits * depth)

        # Build the circuit
        self.circuit = QuantumCircuit(n_qubits)
        # Initial Hadamards
        self.circuit.h(range(n_qubits))
        idx = 0
        for _ in range(depth):
            # Entangling layer: a simple linear chain of CNOTs
            for q in range(n_qubits - 1):
                self.circuit.cx(q, q + 1)
            # Parameterised Ry rotations
            for q in range(n_qubits):
                self.circuit.ry(self.theta[idx], q)
                idx += 1
        # Measure all qubits
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters and return the
        expectation value of Z on qubit 0.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of floats of length ``n_qubits * depth`` that bind to the
            circuit parameters.

        Returns
        -------
        np.ndarray
            1‑D array containing the expectation value.
        """
        # Bind parameters
        param_bind = {self.theta[i]: float(v) for i, v in enumerate(thetas)}
        bound_qc = self.circuit.bind_parameters(param_bind)

        # Run the circuit
        job = execute(bound_qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_qc)

        # Convert counts to expectation value of Z on qubit 0
        exp_val = 0.0
        for bitstring, freq in counts.items():
            # bitstring is little‑endian; last bit corresponds to qubit 0
            z = 1 if bitstring[-1] == "0" else -1
            exp_val += z * freq / self.shots

        return np.array([exp_val])

    def get_parameter_shape(self) -> int:
        """Return the total number of parameters required by the circuit."""
        return self.n_qubits * self.depth


__all__ = ["ParametricFCL"]
