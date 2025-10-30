import numpy as np
import qiskit

def FCL():
    """A simple example of a parameterized quantum circuit for a fully connected layer."""
    class QuantumCircuit:
        """Simple parameterized quantum circuit for demonstration."""

        def __init__(self, n_qubits, backend, shots):
            self._circuit = qiskit.QuantumCircuit(n_qubits)
            self.theta = qiskit.circuit.Parameter("theta")
            self._circuit.h(range(n_qubits))
            self._circuit.barrier()
            self._circuit.ry(self.theta, range(n_qubits))
            self._circuit.measure_all()

            self.backend = backend
            self.shots = shots

        def run(self, thetas):
            job = qiskit.execute(
                self._circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
            result = job.result().get_counts(self._circuit)
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probabilities = counts / self.shots
            expectation = np.sum(states * probabilities)
            return np.array([expectation])
    simulator = qiskit.Aer.get_backend("qasm_simulator")
    circuit = QuantumCircuit(1, simulator, 100)
    return circuit