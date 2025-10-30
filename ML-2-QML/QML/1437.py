import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

class ConvGen404:
    """
    Variational quantum convolution filter.
    Implements a parameterised circuit with RX rotations and a CNOT entangling layer.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.5,
                 backend=None,
                 shots: int = 1024):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Define parameters
        self.theta = [Parameter(f"θ_{i}") for i in range(self.n_qubits)]

        # Build the circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        # Simple entanglement
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Run the circuit on the provided data and return the average
        probability of measuring |1> across all qubits.
        """
        # Flatten data and create parameter binding
        flat = data.reshape(-1)
        param_bind = {self.theta[i]: np.pi if val > self.threshold else 0.0
                      for i, val in enumerate(flat)}

        job = execute(self.circuit,
                      backend=self.backend,
                      shots=self.shots,
                      parameter_binds=[param_bind])
        result = job.result()
        counts = result.get_counts(self.circuit)

        total_ones = 0
        for bitstring, freq in counts.items():
            total_ones += bitstring.count('1') * freq

        return total_ones / (self.shots * self.n_qubits)
