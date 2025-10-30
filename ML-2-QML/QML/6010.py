import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute

class FCL:
    """
    Variational quantum circuit for a fully connected layer.
    The `run` method accepts a list of parameters (thetas) and returns
    a 1‑D NumPy array containing the expectation value of Z on the first qubit.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 1000):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend('qasm_simulator')
        self.circuit = QuantumCircuit(n_qubits)
        # Parameterized rotation
        self.theta = qiskit.circuit.Parameter('theta')
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        # Entanglement if more than one qubit
        if n_qubits > 1:
            for i in range(n_qubits - 1):
                self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, thetas: list[float]) -> np.ndarray:
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas]
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        expectation = 0.0
        for state, count in counts.items():
            z = 1 if state[0] == '0' else -1
            expectation += z * count
        expectation /= self.shots
        return np.array([expectation])

__all__ = ["FCL"]
