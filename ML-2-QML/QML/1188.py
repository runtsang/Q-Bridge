import numpy as np
import qiskit
from qiskit import assemble, transpile

class QuantumCircuit3Q:
    """Parametrised 3‑qubit circuit for a variational quantum expectation."""
    def __init__(self, backend, shots=100):
        self.backend = backend
        self.shots = shots
        self.n_qubits = 3
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta0 = qiskit.circuit.Parameter("theta0")
        self.theta1 = qiskit.circuit.Parameter("theta1")
        self.theta2 = qiskit.circuit.Parameter("theta2")

        # Simple ansatz: H on all, RY on each qubit, CNOT chain
        self.circuit.h(range(self.n_qubits))
        self.circuit.ry(self.theta0, 0)
        self.circuit.ry(self.theta1, 1)
        self.circuit.ry(self.theta2, 2)
        self.circuit.cx(0, 1)
        self.circuit.cx(1, 2)
        self.circuit.measure_all()

        self.compiled = transpile(self.circuit, self.backend)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Run the circuit for an array of parameter vectors.
        Args:
            thetas: shape (batch, 3)
        Returns:
            expectations: shape (batch,)
        """
        expectations = []
        for theta_vec in thetas:
            param_bind = {
                self.theta0: theta_vec[0],
                self.theta1: theta_vec[1],
                self.theta2: theta_vec[2],
            }
            qobj = assemble(self.compiled, parameter_binds=[param_bind], shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            exp = 0.0
            for bitstring, count in counts.items():
                # bitstring[0] corresponds to qubit 0
                bit0 = int(bitstring[0])
                eigen = 1.0 if bit0 == 0 else -1.0
                exp += eigen * count
            exp /= self.shots
            expectations.append(exp)
        return np.array(expectations)

def run_quantum_circuit(params: np.ndarray) -> np.ndarray:
    """Wrapper that accepts a 1‑D array of parameters and returns the
    expectation of Z on qubit 0 for a 3‑qubit ansatz.
    The input is replicated across all three qubits to keep the interface
    simple for the hybrid layer.
    """
    if params.ndim == 1:
        params = np.tile(params[:, None], (1, 3))
    backend = qiskit.Aer.get_backend("aer_simulator")
    qc = QuantumCircuit3Q(backend)
    return qc.run(params)

__all__ = ["QuantumCircuit3Q", "run_quantum_circuit"]
