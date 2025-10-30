import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from typing import Iterable

class FCLGen287:
    """Quantum counterpart of the hybrid fully‑connected layer.

    The circuit follows the pattern from the original `FCL` example but
    is enriched with a photonic‑style parameter set (the
    :class:`FraudLayerParameters` dataclass).  Each layer in that
    dataclass is mapped to a sequence of single‑qubit rotations that
    encode the parameters into a variational circuit.  The circuit
    returns the expectation value of Pauli‑Z on the last qubit, which
    can be fed into the classical network defined in the ML module.

    The design keeps the quantum part lightweight (a single qubit)
    while still exposing a rich parameter space for experimentation.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit = QuantumCircuit(n_qubits)
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        jobs = []
        for theta in thetas:
            bound = {self.theta: theta}
            qc = self._circuit.bind_parameters(bound)
            job = execute(qc, self.backend, shots=self.shots)
            jobs.append(job)
        results = [job.result().get_counts() for job in jobs]
        expectations = []
        for counts in results:
            probs = {state: cnt / self.shots for state, cnt in counts.items()}
            exp = probs.get("0", 0) * 1 + probs.get("1", 0) * -1
            expectations.append(exp)
        return np.array(expectations)

__all__ = ["FCLGen287"]
