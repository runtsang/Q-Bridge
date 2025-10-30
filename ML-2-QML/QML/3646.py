import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit

class HybridConvFCL:
    """
    Quantum implementation of a convolution + fully connected pipeline.
    Uses a parameterized quantum circuit to emulate a convolutional filter
    on a local patch and a second circuit to combine feature expectations
    into a scalar output.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 conv_threshold: float = 127,
                 fc_qubits: int = 4,
                 backend=None,
                 shots: int = 1024) -> None:
        """
        Args:
            kernel_size: Size of the square patch for the convolutional circuit.
            conv_threshold: Threshold on pixel intensity to set rotation angles.
            fc_qubits: Number of qubits in the fully connected circuit.
            backend: Qiskit backend. Defaults to Aer.get_backend('qasm_simulator').
            shots: Number of shots for simulation.
        """
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.fc_qubits = fc_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Build the convolutional circuit template
        self._conv_circuit = self._build_conv_circuit(kernel_size)

        # Build the fully connected circuit template
        self._fc_circuit = self._build_fc_circuit(fc_qubits)

    def _build_conv_circuit(self, n):
        """Creates a parameterized circuit that reads a patch of n² qubits."""
        qc = QuantumCircuit(n)
        theta = [Parameter(f"theta{i}") for i in range(n)]
        for i in range(n):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n, 2)
        qc.measure_all()
        return qc

    def _build_fc_circuit(self, m):
        """Creates a parameterized circuit that consumes m expectation values."""
        qc = QuantumCircuit(m)
        theta = Parameter("theta")  # shared parameter for demo
        qc.h(range(m))
        qc.barrier()
        qc.ry(theta, range(m))
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the hybrid quantum pipeline.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape (kernel_size, kernel_size). Pixel values in [0,255].

        Returns
        -------
        float
            Scalar output obtained as expectation of the fully connected circuit.
        """
        # Flatten and bind parameters for convolution circuit
        flat = np.reshape(data, (1, self.kernel_size ** 2))
        param_binds = []
        for row in flat:
            bind = {}
            for i, val in enumerate(row):
                bind[self._conv_circuit.parameters[i]] = np.pi if val > self.conv_threshold else 0.0
            param_binds.append(bind)

        # Execute conv circuit
        job_conv = execute(
            self._conv_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds
        )
        result_conv = job_conv.result().get_counts(self._conv_circuit)
        # Compute expectation value per qubit: average of |1> probability
        expectations = []
        for key, cnt in result_conv.items():
            prob1 = cnt / self.shots
            for i, bit in enumerate(reversed(key)):
                if bit == '1':
                    expectations.append(prob1)
        if not expectations:
            expectations = [0.0] * self.kernel_size ** 2

        # Prepare fully connected circuit
        theta_val = np.mean(expectations) * np.pi
        bind_fc = {self._fc_circuit.parameters[0]: theta_val}

        job_fc = execute(
            self._fc_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind_fc]
        )
        result_fc = job_fc.result().get_counts(self._fc_circuit)
        total = 0
        for state, cnt in result_fc.items():
            num_ones = sum(int(b) for b in state)
            total += num_ones * cnt
        expectation = total / (self.shots * self.fc_qubits)
        return expectation

__all__ = ["HybridConvFCL"]
