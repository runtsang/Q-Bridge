import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter

class ConvQuantum:
    """
    Quantum circuit for a 2D convolution filter. It processes a flattened
    2D patch and returns the average probability of measuring |1> across
    all qubits. The circuit is parameterized by a set of RX gates and
    entangling CNOTs.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, device=None):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

        # Build circuit
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.measure_all()

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            data: numpy array of shape (batch_size, kernel_size, kernel_size)

        Returns:
            numpy array of shape (batch_size,) with average |1> probability.
        """
        batch_size = data.shape[0]
        flat = data.reshape(batch_size, self.n_qubits)
        probs = []
        for patch in flat:
            bind = {}
            for idx, val in enumerate(patch):
                bind[self.theta[idx]] = np.pi if val > self.threshold else 0.0
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
            result = job.result()
            counts = result.get_counts(self.circuit)
            prob_ones = 0.0
            total = sum(counts.values())
            for outcome, cnt in counts.items():
                ones_in_outcome = sum(int(b) for b in outcome)
                prob_ones += (ones_in_outcome / self.n_qubits) * (cnt / total)
            probs.append(prob_ones)
        return np.array(probs)
