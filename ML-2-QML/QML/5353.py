import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp

class QuanvCircuit:
    """Quantum convolution filter for 2×2 data patches."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = ParameterVector("theta", self.n_qubits)
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Run the filter and return average probability of measuring |1>."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = self.backend.run(assemble(transpile(self._circuit, self.backend),
                                        parameter_binds=param_binds,
                                        shots=self.shots))
        result = job.result().get_counts()
        counts = 0
        for key, val in result.items():
            ones = sum(int(b) for b in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

def build_classifier_circuit(num_qubits: int, depth: int):
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        circuit.rx(encoding[i], i)

    idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            circuit.ry(weights[idx], i)
            idx += 1
        for i in range(num_qubits - 1):
            circuit.cz(i, i + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, encoding, weights, observables

class QuantumCircuitWrapper:
    """Simple parameterised two‑qubit circuit for expectation evaluation."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.backend = backend
        self.shots = shots
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = ParameterVector("theta", n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=[{self.theta[i]: theta for i, theta in enumerate(thetas)}])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            probs = np.array(list(count_dict.values())) / self.shots
            states = np.array([int(k, 2) for k in count_dict.keys()])
            return np.dot(states, probs)
        return np.array([expectation(result)])

class Hybrid:
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        self.quantum = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute expectation value for each input sample."""
        return self.quantum.run(inputs + self.shift)

class HybridSamplerQNN:
    """
    Quantum sampler network that emulates the hybrid architecture.
    Consists of a quantum convolution filter, a quantum expectation head
    and a sampler that returns a probability distribution.
    """
    def __init__(self, kernel_size: int = 2, shots: int = 200,
                 threshold: float = 0.5, shift: float = np.pi/2):
        self.backend = AerSimulator()
        self.filter = QuanvCircuit(kernel_size, self.backend, shots, threshold)
        self.hybrid = Hybrid(kernel_size**2, self.backend, shots, shift)
        self.sampler = self.backend  # using the backend as a sampler

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: 2‑D array of shape (kernel_size, kernel_size)
        Returns:
            probs: numpy array of shape (2,) with class probabilities
        """
        # 1. Convolution filter measurement
        filt_prob = self.filter.run(data)  # scalar

        # 2. Expectation head
        expect = self.hybrid.forward(np.array([filt_prob]))[0]  # scalar

        # 3. Generate categorical probabilities
        probs = np.array([expect, 1 - expect])
        return probs

    def sample(self, data: np.ndarray, n_samples: int = 10) -> np.ndarray:
        probs = self.forward(data)
        return np.random.choice([0,1], size=n_samples, p=probs)

__all__ = ["HybridSamplerQNN"]
