"""Quantum version of a fully‑connected layer using a parameterized Ry circuit."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from typing import Sequence

class FCL:
    """Parameterized quantum circuit that emulates a fully‑connected layer.

    The circuit consists of `depth` layers of Ry rotations followed by a linear
    entanglement using CNOT gates.  The `run` method evaluates the expectation
    value of the Pauli‑Z operator for each qubit.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        depth: int = 1,
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        theta = self.theta
        for d in range(self.depth):
            # Parameterized Ry on each qubit
            for i in range(self.n_qubits):
                qc.ry(theta, i)
            # Entangle qubits with a linear chain of CNOTs
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            qc.barrier()
        qc.measure_all()
        return qc

    def run(self, thetas: Sequence[float]) -> np.ndarray:
        """
        Evaluate the circuit for a given set of rotation angles.

        Parameters
        ----------
        thetas : Sequence[float]
            Rotation angles for each qubit.  Length must equal `n_qubits`.

        Returns
        -------
        np.ndarray
            Expectation values ⟨Z⟩ for each qubit.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError("Length of `thetas` must match `n_qubits`.")
        # Bind the same parameter to each qubit independently
        param_binds = [{self.theta: theta} for theta in thetas]
        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self._circuit)
        expectation = np.zeros(self.n_qubits)
        for outcome, count in counts.items():
            bits = np.array([1 if b == "0" else -1 for b in outcome[::-1]])
            expectation += count * bits
        expectation /= self.shots
        return expectation
