import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class UnifiedQuantumHybridLayer:
    """
    Quantum‑only implementation that mirrors the hybrid structure of the
    classical module but operates entirely on qubits.
    """
    def __init__(self, num_qubits: int, depth: int, shots: int = 100, threshold: float = 127):
        """
        Build a layered ansatz with data‑encoding and variational Ry gates.
        """
        # Data‑encoding parameters
        self.encoding = ParameterVector("x", num_qubits)
        # Variational parameters
        self.theta = ParameterVector("theta", num_qubits * depth)

        self.circuit = QuantumCircuit(num_qubits)
        # Encode classical data with RX gates
        for i, p in enumerate(self.encoding):
            self.circuit.rx(p, i)

        # Layered variational ansatz
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                self.circuit.ry(self.theta[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)

        # Final measurement
        self.circuit.measure_all()

        self.observable = SparsePauliOp("Z" * num_qubits)
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the circuit on a 2‑D array of shape (kernel_size, kernel_size).
        Data is first encoded into RX parameters, then the variational ansatz
        is applied and the expectation of Z on the last qubit is returned.
        """
        flat = data.flatten()
        param_binds = []
        for i, val in enumerate(flat):
            bind = {self.encoding[i]: np.pi if val > self.threshold else 0}
            param_binds.append(bind)

        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result()

        counts = result.get_counts(self.circuit)
        exp = 0.0
        for bitstring, cnt in counts.items():
            # Use the last qubit's Z eigenvalue (+1/-1)
            z_val = 1 if bitstring[-1] == '0' else -1
            exp += z_val * cnt
        exp /= self.shots
        return np.array([exp])

__all__ = ["UnifiedQuantumHybridLayer"]
