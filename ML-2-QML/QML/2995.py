import numpy as np
import qiskit
from qiskit import QuantumCircuit as QC, Aer, transpile, assemble

class QuantumCircuit:
    """
    Parameterised two‑qubit circuit used as a quantum expectation head.
    The circuit prepares a uniform superposition, applies a rotation
    Ry(θ) to each qubit, measures in the computational basis,
    and returns the expectation value of the Z observable of each qubit.
    """
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")

        self.circuit = QC(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit with the given rotation angles.
        `thetas` is a 1‑D array of shape (n_qubits,).
        Returns an array containing the expectation value for each qubit.
        """
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        exp_vals = []
        for q in range(self.n_qubits):
            # Expectation value of Z for qubit q
            counts = result
            exp_z = 0.0
            for state, count in counts.items():
                if int(state[-(q+1)]) == 0:
                    exp_z += count
                else:
                    exp_z -= count
            exp_z /= self.shots
            exp_vals.append(exp_z)
        return np.array(exp_vals)

class HybridQuantumFullyConnectedClassifier:
    """
    Hybrid fully‑connected quantum layer.
    The forward pass runs a parameterised quantum circuit for each
    input feature vector and feeds the resulting expectation values
    into a sigmoid activation to produce a probability distribution.
    """
    def __init__(self, n_features: int, n_qubits: int = 2, shift: float = 0.0, shots: int = 1024):
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.shift = shift
        self.qc = QuantumCircuit(n_qubits, shots=shots)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the quantum expectation for each input vector.
        `x` is a 2‑D array of shape (batch_size, n_features).
        Returns a 2‑D array of shape (batch_size, 2) containing
        probabilities for the two classes.
        """
        probs = []
        for vec in x:
            exp = self.qc.run(vec[:self.n_qubits])
            logits = exp.sum()  # simple aggregation
            prob = 1 / (1 + np.exp(-(logits + self.shift)))
            probs.append([prob, 1 - prob])
        return np.array(probs)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Public interface mirroring the classical version.
        Accepts a vector of rotation angles and returns a probability array.
        """
        exp = self.qc.run(thetas[:self.n_qubits])
        logits = exp.sum()
        prob = 1 / (1 + np.exp(-(logits + self.shift)))
        return np.array([prob, 1 - prob])
