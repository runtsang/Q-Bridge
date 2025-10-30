import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

class ConvGen194:
    """
    Quantum convolution filter with parameterized entanglement depth.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 backend=None,
                 shots: int = 100,
                 threshold: float = 0.0,
                 entanglement_depth: int = 2):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend('qasm_simulator')
        self.entanglement_depth = entanglement_depth
        self.theta = ParameterVector('theta', self.n_qubits)
        self._build_circuit()

    def _build_circuit(self):
        self.circuit = QuantumCircuit(self.n_qubits)
        # Parameterized RX gates
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        # Entanglement layers
        for _ in range(self.entanglement_depth):
            for i in range(self.n_qubits - 1):
                self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum circuit on 2D data and return the average probability of measuring |1>.
        """
        data = np.reshape(data, (self.n_qubits,))
        # Bind parameters based on threshold
        param_binds = []
        for val in data:
            bind = {}
            for i in range(self.n_qubits):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts()
        prob = 0.0
        for bitstring, count in counts.items():
            ones = bitstring.count('1')
            prob += ones * count
        prob /= (self.shots * self.n_qubits)
        return prob

def Conv(*args, **kwargs):
    """
    Convenience function that returns a ConvGen194 instance.
    """
    return ConvGen194(*args, **kwargs)
