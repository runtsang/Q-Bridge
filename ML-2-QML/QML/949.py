import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

class HybridFCL:
    """
    Parameterized quantum circuit that implements a fully connected layer.
    The circuit applies Ry rotations to each qubit followed by a CNOT ladder
    to entangle them. The expectation value of Z on each qubit is returned.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.theta = [Parameter(f"theta_{i}") for i in range(n_qubits)]
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        qc.barrier()
        for i, th in enumerate(self.theta):
            qc.ry(th, i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i+1)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, thetas):
        """
        thetas: array-like of length n_qubits
        Returns: np.ndarray of shape (n_qubits,)
        """
        bound_circuit = self.circuit.bind_parameters(
            {th: val for th, val in zip(self.theta, thetas)}
        )
        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)
        probs = {state: count for state, count in counts.items()}
        expectation = []
        for qubit in range(self.n_qubits):
            exp_z = 0.0
            for state, count in probs.items():
                bit = int(state[-(qubit+1)])
                exp_z += ((-1)**bit) * count
            exp_z /= self.shots
            expectation.append(exp_z)
        return np.array(expectation)

__all__ = ["HybridFCL"]
