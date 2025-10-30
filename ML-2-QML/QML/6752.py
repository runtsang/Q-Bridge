import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import execute

class ConvGen611:
    """
    Quantum quanvolution filter with optional regression head.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 127, shots: int = 100,
                 backend=None, regression: bool = False, regression_weights=None):
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self._circuit = self._build_circuit()
        self.regression = regression
        if regression:
            if regression_weights is None:
                self.weights = np.random.randn(self.n_qubits, 1).astype(np.float32)
                self.bias = np.random.randn(1).astype(np.float32)
            else:
                self.weights, self.bias = regression_weights

    def _build_circuit(self):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(theta):
            circ.rx(p, i)
        circ.barrier()
        circ += random_circuit(self.n_qubits, 2)
        circ.measure_all()
        return circ

    def _bind_params(self, data_flat):
        param_binds = []
        for val in data_flat:
            bind = {}
            for i, v in enumerate(val):
                bind[self._circuit.parameters[i]] = np.pi if v > self.threshold else 0
            param_binds.append(bind)
        return param_binds

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        data: 2D array shape (kernel, kernel)
        Returns per‑qubit probability of measuring |1>.
        """
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = self._bind_params(data_flat)
        job = execute(self._circuit, self.backend, shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts_per_qubit = np.zeros(self.n_qubits)
        for key, val in result.items():
            for i, bit in enumerate(key):
                if bit == "1":
                    counts_per_qubit[i] += val
        probs = counts_per_qubit / (self.shots * len(result))
        return probs

    def predict(self, data: np.ndarray) -> float:
        """
        If regression flag is set, returns weighted sum of qubit expectation values.
        """
        probs = self.run(data)
        return float(probs @ self.weights + self.bias)

def generate_superposition_data(num_wires: int, samples: int):
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset:
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": np.array(self.states[idx], dtype=np.complex64),
            "target": np.array(self.labels[idx], dtype=np.float32)
        }

__all__ = ["ConvGen611", "generate_superposition_data", "RegressionDataset"]
