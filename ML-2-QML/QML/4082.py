import numpy as np
import qiskit
from qiskit import transpile, assemble
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.aer import AerSimulator

class QuantumHybridConvFilter:
    """
    Quantum convolutional filter that encodes a 2‑D kernel patch into qubits
    and returns the average probability of measuring |1> across all qubits.
    """
    def __init__(self, kernel_size: int = 2, backend=None,
                 shots: int = 1024, threshold: float = 0.5):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.threshold = threshold

        # Build variational circuit
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        # Encode each pixel with an RX rotation
        self.params = [qiskit.circuit.Parameter(f'theta{i}') for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.params[i], i)
        # Entanglement layer
        self.circuit.h(range(self.n_qubits))
        self.circuit.barrier()
        # Variational ansatz
        self.circuit += RealAmplitudes(self.n_qubits, reps=3)
        self.circuit.measure_all()

    def run(self, data):
        """
        Execute the quantum filter on a 2‑D kernel patch.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size)

        Returns:
            float: average probability of measuring |1> across all qubits.
        """
        data = np.asarray(data).reshape(self.n_qubits)
        param_binds = [{p: np.pi if val > self.threshold else 0}
                       for p, val in zip(self.params, data)]
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result().get_counts(self.circuit)

        # Compute average probability of measuring |1>
        prob_one = 0.0
        for bitstring, count in result.items():
            ones = bitstring.count('1')
            prob_one += ones * count
        prob_one /= (self.shots * self.n_qubits)
        return prob_one

def Conv(**kwargs):
    """
    Factory that returns a QuantumHybridConvFilter instance.
    """
    return QuantumHybridConvFilter(**kwargs)
