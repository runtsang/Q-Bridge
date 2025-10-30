import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Pauli
from typing import Iterable

class FCL:
    """
    Quantum fully‑connected layer implemented as a parameterized variational
    circuit. The circuit consists of an entangling CX layer and Ry rotations
    with trainable angles. Exact expectation values are computed using a
    state‑vector simulator.

    Example:
        >>> qcl = FCL(n_qubits=2)
        >>> qcl.run([0.1, 0.5])
    """

    def __init__(self, n_qubits: int, backend=None) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.params = [Parameter(f"θ_{i}") for i in range(n_qubits)]
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Entangling layer between adjacent qubits
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        # Parameterized Ry rotations
        for i, p in enumerate(self.params):
            qc.ry(p, i)
        # Measurement of all qubits (required for the state‑vector simulator)
        qc.measure_all()
        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the expectation value of Z on the first qubit for the
        given rotation angles.

        Parameters
        ----------
        thetas : Iterable[float]
            Rotation angles for each qubit.

        Returns
        -------
        np.ndarray
            Array containing the expectation value.
        """
        bound_circuit = self.circuit.bind_parameters(
            {param: theta for param, theta in zip(self.params, thetas)}
        )
        state = Statevector.from_instruction(bound_circuit)
        expectation = state.expectation_value(
            Pauli("Z" + "I" * (self.n_qubits - 1))
        )
        return np.array([expectation])

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying parameterized circuit."""
        return self.circuit

__all__ = ["FCL"]
