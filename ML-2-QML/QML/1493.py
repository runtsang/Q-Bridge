import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator

class FCL:
    """
    Variational fully connected layer implemented as a parameterized quantum circuit.
    The circuit comprises `depth` layers of CNOT entanglement followed by Ry rotations.
    The `run` method executes the circuit for a single set of parameters and returns
    the expectation of the binary measurement (interpreted as a real number).
    """

    def __init__(self,
                 n_qubits: int = 1,
                 depth: int = 2,
                 backend=None,
                 shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or AerSimulator()
        self.circuit = QuantumCircuit(n_qubits)
        self.params = []
        self._build_circuit()

    def _build_circuit(self):
        """Construct the layered entangled circuit with Ry rotations."""
        for d in range(self.depth):
            # Entangling layer
            for q in range(self.n_qubits - 1):
                self.circuit.cx(q, q + 1)
            # Parameterized Ry rotations
            for q in range(self.n_qubits):
                theta = qiskit.circuit.Parameter(f"theta_{d}_{q}")
                self.params.append(theta)
                self.circuit.ry(theta, q)
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for the supplied parameter vector.
        Returns a NumPy array containing the expectation value of the measurement.
        """
        if len(thetas)!= len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} parameters, got {len(thetas)}"
            )
        param_bind = {p: t for p, t in zip(self.params, thetas)}
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array([int(k, 2) for k in result.keys()])
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def get_params(self) -> int:
        """Return the total number of trainable parameters."""
        return len(self.params)

__all__ = ["FCL"]
