import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from typing import Iterable, Sequence

class QuantumVariationalLayer:
    """
    Parameterized quantum circuit with configurable depth.  Each layer
    applies Rx, Ry, Rz rotations followed by a linear chain of CNOTs.
    The circuit is fully parameterized and compatible with Qiskit's
    simulation back‑end.
    """
    def __init__(
        self,
        n_qubits: int = 1,
        n_layers: int = 2,
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.theta = ParameterVector("theta", length=self.n_qubits * self.n_layers * 3)
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        idx = 0
        for _ in range(self.n_layers):
            # Single‑qubit rotations
            qc.rx(self.theta[idx], range(self.n_qubits)); idx += self.n_qubits
            qc.ry(self.theta[idx], range(self.n_qubits)); idx += self.n_qubits
            qc.rz(self.theta[idx], range(self.n_qubits)); idx += self.n_qubits
            # Entanglement
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            qc.barrier()
        qc.measure_all()
        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters and return the
        expectation value of the computational basis (interpreted as a
        real number).  The interface mirrors the classical `run` method.
        """
        # Build parameter binding dictionary
        param_dict = {self.theta[i]: val for i, val in enumerate(thetas)}
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_dict],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        expectation = np.sum(states * probs)
        return np.array([expectation])

def FCL() -> QuantumVariationalLayer:
    """
    Public factory returning the variational quantum layer instance.
    """
    return QuantumVariationalLayer()

__all__ = ["FCL"]
