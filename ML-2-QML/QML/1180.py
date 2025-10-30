import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter

class FCL:
    """
    Variational quantum circuit that mimics a fully‑connected layer.
    The circuit consists of ``n_layers`` entangling layers of
    parameterised Ry rotations followed by a chain of CNOTs.
    ``run`` accepts a flat list of parameters and returns the
    expectation values of the Pauli‑Z observable on each qubit.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        # Parameter placeholders
        self.params = [
            [Parameter(f"theta_{l}_{q}") for q in range(n_qubits)]
            for l in range(n_layers)
        ]
        self._build_circuit()

    def _build_circuit(self) -> None:
        self.circuit = QuantumCircuit(self.n_qubits)
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                self.circuit.ry(self.params[l][q], q)
            # Entangle neighbours
            for q in range(self.n_qubits - 1):
                self.circuit.cx(q, q + 1)
            self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Bind the supplied parameters to the circuit, execute it and
        return the expectation values of Z on each qubit as a 1‑D
        numpy array.
        """
        if len(thetas)!= self.n_qubits * self.n_layers:
            raise ValueError(
                f"Expected {self.n_qubits * self.n_layers} parameters, got {len(thetas)}"
            )
        # Build the binding dictionary
        bind_dict = {}
        idx = 0
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                bind_dict[self.params[l][q]] = thetas[idx]
                idx += 1
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind_dict],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        expectation = []
        for q in range(self.n_qubits):
            exp = 0.0
            for state, cnt in counts.items():
                # Qiskit returns bitstrings with qubit 0 as the leftmost bit
                bit = int(state[::-1][q])  # little‑endian
                exp += ((-1) ** bit) * cnt
            exp /= self.shots
            expectation.append(exp)
        return np.array(expectation)
