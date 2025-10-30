import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute

class FCL:
    """
    Parameterized quantum circuit for a fully connected layer with entanglement.
    The circuit uses 2 qubits, 3 rotation parameters per qubit, and a CNOT
    entangling gate.  The ``run`` method executes the circuit and returns the
    expectation value of the Pauli‑Z operator on the first qubit.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend('qasm_simulator')

        # Parameter placeholders
        self.theta = [qiskit.circuit.Parameter(f'theta_{i}') for i in range(n_qubits * 3)]

        # Build the circuit
        self.circuit = QuantumCircuit(n_qubits)
        # Input encoding: Hadamard on each qubit
        for i in range(n_qubits):
            self.circuit.h(i)
        self.circuit.barrier()

        # Variational layer: 3 rotations per qubit
        for i in range(n_qubits):
            self.circuit.ry(self.theta[i], i)
            self.circuit.rz(self.theta[i + n_qubits], i)
            self.circuit.rx(self.theta[i + 2 * n_qubits], i)

        # Entanglement
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)

        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with a list of parameters (length 3 * n_qubits)
        and return the expectation value of Pauli‑Z on the first qubit.
        """
        if len(thetas)!= len(self.theta):
            raise ValueError(f"Expected {len(self.theta)} parameters, got {len(thetas)}.")

        param_dict = {self.theta[i]: val for i, val in enumerate(thetas)}
        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[param_dict])
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Convert counts to probabilities and compute expectation
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(bitstring, 2) for bitstring in counts.keys()])
        # Pauli‑Z expectation on first qubit: map bit 0->+1, bit 1->-1
        z_values = 1 - 2 * ((states >> (self.n_qubits - 1)) & 1)
        expectation = np.sum(z_values * probs)

        return np.array([expectation])

__all__ = ["FCL"]
