import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit import execute, Aer

class QuanvCircuit:
    """Quantum analogue of a convolutional filter using a random circuit."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        """Return the mean probability of measuring |1> across all qubits."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class SamplerQNN:
    """Parameterized quantum sampler circuit producing a 2‑class distribution."""
    def __init__(self, input_params, weight_params):
        self.input_params = input_params
        self.weight_params = weight_params
        self.circuit = qiskit.QuantumCircuit(2)
        self.circuit.ry(input_params[0], 0)
        self.circuit.ry(input_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(weight_params[0], 0)
        self.circuit.ry(weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(weight_params[2], 0)
        self.circuit.ry(weight_params[3], 1)
        self.circuit.measure_all()

class HybridSamplerConv:
    """Quantum‑classical hybrid sampler that first evaluates a QuanvCircuit and then samples."""
    def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 127):
        self.backend = Aer.get_backend("qasm_simulator")
        self.conv_circuit = QuanvCircuit(kernel_size, self.backend, shots, threshold)
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)
        self.sampler_qnn = SamplerQNN(self.input_params, self.weight_params)

    def run(self, data):
        """
        Parameters
        ----------
        data : np.ndarray
            Input window of shape (kernel_size, kernel_size).

        Returns
        -------
        dict
            Probabilities for each 2‑bit outcome.
        """
        conv_prob = self.conv_circuit.run(data)
        # Use conv output as one sampler input; fix second input to 0.5
        bind = {self.input_params[0]: conv_prob, self.input_params[1]: 0.5}
        job = execute(self.sampler_qnn.circuit, self.backend, shots=100, parameter_binds=[bind])
        result = job.result().get_counts(self.sampler_qnn.circuit)
        total = sum(result.values())
        probs = {k: v / total for k, v in result.items()}
        return probs

__all__ = ["HybridSamplerConv"]
