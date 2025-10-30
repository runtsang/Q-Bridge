"""Quantum variational fully‑connected layer using Qiskit.

The class builds a parameterized circuit that applies a Ry rotation to each qubit
followed by a CNOT ladder to create entanglement.  The circuit measures all qubits
and returns the expectation value of the Z operator summed over all qubits.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from typing import Iterable

class FCL:
    """
    Variational quantum circuit that serves as a fully‑connected layer.
    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    shots : int, optional
        Number of shots for measurement.
    backend : qiskit.providers.backend.Backend, optional
        Backend to run the circuit on.  Defaults to Aer qasm_simulator.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 1024,
                 backend: qiskit.providers.backend.Backend = None) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._build_circuit()

    def _build_circuit(self) -> None:
        """
        Build a parameterized circuit with Ry rotations and a CNOT ladder.
        """
        self.circuit = QuantumCircuit(self.n_qubits)
        # Parameter for each qubit
        self.theta = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        self.circuit.h(range(self.n_qubits))
        for i, t in enumerate(self.theta):
            self.circuit.ry(t, i)
        # Entanglement
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the given parameters and return the expectation value.
        Parameters
        ----------
        thetas : Iterable[float]
            Parameters for the Ry gates, one per qubit.
        Returns
        -------
        np.ndarray
            Expectation value of the sum of Z measurements.
        """
        param_bindings = [{self.theta[i]: theta} for i, theta in enumerate(thetas)]
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_bindings,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        # Convert measurement outcomes to expectation value of Z
        exp = 0.0
        for state, count in counts.items():
            # Interpret state string as binary bits
            z = [1 if bit == "0" else -1 for bit in state[::-1]]  # reverse order
            exp += sum(z) * count
        exp /= self.shots
        return np.array([exp])

__all__ = ["FCL"]
