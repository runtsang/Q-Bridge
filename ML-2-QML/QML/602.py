import numpy as np
import qiskit
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.providers import Backend
from qiskit import Aer, execute

class FCLGenQuantum:
    """
    Variational quantum circuit mimicking a fully connected layer.

    Features:
    * Support for multiple qubits and circuit depth.
    * Parameterised Ry rotations per qubit.
    * Entangling CNOT chain to introduce correlations.
    * Expectation value of Pauli‑Z on the first qubit as output.
    """
    def __init__(self, n_qubits: int = 1, depth: int = 1, backend: Backend | None = None, shots: int = 1000):
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.theta = Parameter("θ")
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for _ in range(self.depth):
            # single‑qubit Ry rotations
            qc.ry(self.theta, range(self.n_qubits))
            # entangle neighbouring qubits
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with a list of parameter values.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameter values to bind to the single Ry rotation parameter.
            The length must match the number of shots or be broadcasted.
        """
        # bind the same theta to each parameter instance
        param_binds = [{self.theta: theta} for theta in thetas]
        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self._circuit)
        # compute expectation value of Z on first qubit
        exp = 0.0
        for state, cnt in counts.items():
            # state string is little‑endian; first qubit is last char
            bit = int(state[-1])
            z = 1.0 if bit == 0 else -1.0
            exp += z * cnt
        exp /= self.shots
        return np.array([exp])

__all__ = ["FCLGenQuantum"]
