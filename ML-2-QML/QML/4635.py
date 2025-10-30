import numpy as np
import qiskit

class FCL:
    """
    Quantum implementation of the hybrid fully‑connected layer.  A variational
    circuit with a random entangling layer is executed on a simulator or
    real device.
    """
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._build_circuit()

    def _build_circuit(self):
        theta_params = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        qc = qiskit.QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        for i, param in enumerate(theta_params):
            qc.ry(param, i)
        # Random entangling layer (8 CX gates)
        for _ in range(8):
            q, p = np.random.choice(self.n_qubits, 2, replace=False)
            qc.cx(q, p)
        qc.measure_all()
        self.qc = qc
        self.theta_params = theta_params

    def run(self, thetas):
        param_bind = {self.theta_params[i]: t for i, t in enumerate(thetas)}
        job = qiskit.execute(self.qc, self.backend, shots=self.shots,
                             parameter_binds=[param_bind])
        result = job.result()
        counts = result.get_counts()
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.sum(states * probs) / (2**self.n_qubits - 1)
        return np.array([expectation])

__all__ = ["FCL"]
