import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import ParameterVector

class ConvGen118Q:
    """Variational quantum filter that mimics the classical ConvGen118.

    The circuit is a parameterised ansatz with a trainable rotation angle per qubit.
    It supports a simple depth‑wise entanglement pattern and can be trained using
    the parameter‑shift rule.  The `run` method returns the average probability of
    measuring |1> across all qubits for a given image patch.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 1024,
        depth: int = 2,
        threshold: float = 0.5,
        learning_rate: float = 0.01,
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self.kernel_size = kernel_size
        self.shots = shots
        self.threshold = threshold
        self.learning_rate = learning_rate

        # Backend
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Data‑encoding parameters
        self.data_params = ParameterVector("data", self.n_qubits)
        # Ansatz parameters
        self.theta = ParameterVector("theta", self.n_qubits)
        # Current numeric values of the ansatz parameters
        self.params = np.zeros(self.n_qubits, dtype=np.float32)

        # Build the circuit
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.rx(self.data_params[i], i)
        for _ in range(depth):
            for i in range(self.n_qubits - 1):
                self.circuit.cx(i, i + 1)
            self.circuit.barrier()
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.measure_all()

    def _bind_params(self, data):
        """Create a binding dictionary for both data encoding and ansatz."""
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        data = data.reshape(-1)
        bind = {
            self.data_params[i]: np.pi if val > self.threshold else 0.0
            for i, val in enumerate(data)
        }
        bind.update({self.theta[i]: self.params[i] for i in range(self.n_qubits)})
        return bind

    def run(self, data):
        """Execute the circuit for the given image patch and return average |1> probability."""
        bind = self._bind_params(data)
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        total_ones = sum(freq * sum(int(b) for b in bitstring) for bitstring, freq in counts.items())
        return total_ones / (self.shots * self.n_qubits)

    def _eval_shifted(self, data, params):
        """Helper to evaluate the circuit with a specific set of ansatz parameters."""
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        data = data.reshape(-1)
        bind = {
            self.data_params[i]: np.pi if val > self.threshold else 0.0
            for i, val in enumerate(data)
        }
        bind.update({self.theta[i]: params[i] for i in range(self.n_qubits)})
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        total_ones = sum(freq * sum(int(b) for b in bitstring) for bitstring, freq in counts.items())
        return total_ones / (self.shots * self.n_qubits)

    def fit(self, data, target, epochs: int = 1):
        """Update the ansatz parameters using the parameter‑shift rule."""
        for _ in range(epochs):
            grads = np.zeros(self.n_qubits, dtype=np.float32)
            for i in range(self.n_qubits):
                shift = np.pi / 2
                params_plus = self.params.copy()
                params_minus = self.params.copy()
                params_plus[i] += shift
                params_minus[i] -= shift
                exp_plus = self._eval_shifted(data, params_plus)
                exp_minus = self._eval_shifted(data, params_minus)
                grads[i] = (exp_plus - exp_minus) / 2.0
            # Gradient descent step
            self.params -= self.learning_rate * grads

def Conv():
    """Factory that returns a ConvGen118Q instance with default settings."""
    return ConvGen118Q()
