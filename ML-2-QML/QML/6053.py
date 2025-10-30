import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import Aer, execute

class ConvEnhanced:
    """
    Quantum convolution filter that mirrors the classical Conv filter.
    The filter uses a parameterized RX circuit on a grid of qubits
    corresponding to the input patch. The parameters are bound to the
    input data and the measurement expectation is returned.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 backend=None,
                 shots: int = 1024,
                 threshold: float = 0.5):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self._build_circuit()

    def _build_circuit(self):
        """Build a base circuit with RX rotations for each qubit."""
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        for i, t in enumerate(self.theta):
            self.circuit.rx(t, i)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum circuit on classical data.

        Args:
            data: 2D array with shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                # Map input value to rotation angle in [0, π]
                bind[self.theta[i]] = np.pi * (val > self.threshold)
            param_binds.append(bind)

        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)

        total_ones = 0
        for key, val in counts.items():
            ones = key.count('1')
            total_ones += ones * val

        expectation = total_ones / (self.shots * self.n_qubits)
        return expectation

    def __call__(self, data: np.ndarray) -> float:
        """Allow the instance to be called directly."""
        return self.run(data)

def Conv():
    """Convenience factory that returns an instance of ConvEnhanced."""
    return ConvEnhanced()
