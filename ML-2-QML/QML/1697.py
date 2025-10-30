import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import ParameterVector

class ConvHybrid:
    """
    Quantum variational filter that can be used as a drop‑in replacement for the original Conv.

    Features:
        * Parameterized Ry rotation per qubit (trainable theta)
        * Data‑driven encoding via Ry with angle pi if pixel > threshold, else 0
        * Simple entanglement: CNOT chain between consecutive qubits
        * Measures all qubits; returns average probability of measuring |1>

    The API mirrors the original: ConvHybrid().run(data) -> float
    """

    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Parameter vectors for trainable weights and data encoding
        self.theta = ParameterVector("theta", length=self.n_qubits)
        self.data = ParameterVector("data", length=self.n_qubits)

        # Build the variational circuit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self._circuit.ry(self.theta[i], i)
        self._circuit.barrier()
        for i in range(self.n_qubits):
            self._circuit.ry(self.data[i], i)
        self._circuit.barrier()
        for i in range(self.n_qubits - 1):
            self._circuit.cx(i, i + 1)
        self._circuit.barrier()
        self._circuit.measure_all()

    def run(self, data) -> float:
        """
        Run the quantum filter on a 2‑D patch.

        Args:
            data: 2‑D array or list of shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across all qubits.
        """
        flat = np.array(data, dtype=np.float64).reshape(-1)

        # Bind data encoding parameters
        data_bindings = {self.data[i]: (np.pi if val > self.threshold else 0.0) for i, val in enumerate(flat)}

        # Bind trainable theta parameters
        if hasattr(self, "theta_vals"):
            theta_bindings = {self.theta[i]: val for i, val in enumerate(self.theta_vals)}
        else:
            theta_bindings = {self.theta[i]: 0.0 for i in range(self.n_qubits)}

        param_binds = {**data_bindings, **theta_bindings}

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_binds]
        )
        result = job.result()
        counts = result.get_counts(self._circuit)

        total_ones = 0
        total_counts = 0
        for bitstring, count in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * count
            total_counts += count

        avg_prob = total_ones / (total_counts * self.n_qubits)
        return avg_prob

    def set_theta(self, theta_vals):
        """
        Set the values of the trainable parameters.

        Args:
            theta_vals: list or array of length n_qubits.
        """
        if len(theta_vals)!= self.n_qubits:
            raise ValueError("theta_vals must have length equal to number of qubits")
        self.theta_vals = np.array(theta_vals)

    def get_theta(self):
        """
        Return current values of trainable parameters.
        """
        return getattr(self, "theta_vals", np.zeros(self.n_qubits))
