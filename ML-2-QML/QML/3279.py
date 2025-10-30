import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

class QuantumFC:
    """Quantum fully‑connected layer implemented as a parameterized Ry circuit."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

class QuantumConv:
    """Quantum convolution filter (quanvolution) using a random two‑layer circuit."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: int):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
        ]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class HybridLayer:
    """
    Hybrid quantum layer that applies a quanvolution filter followed by a quantum fully‑connected layer.
    Only the 'qml' mode is supported in this module; the classical counterpart is provided in ml_code.
    """
    def __init__(self, mode: str = 'qml', n_qubits: int = 1,
                 kernel_size: int = 2, threshold: int = 127, shots: int = 100) -> None:
        if mode!= 'qml':
            raise ValueError("Only 'qml' mode is supported in the quantum module.")
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.conv = QuantumConv(kernel_size, backend, shots, threshold)
        self.fc = QuantumFC(n_qubits, backend, shots)

    def run(self, data):
        """
        Run the data through the quantum convolution filter and then the quantum fully‑connected layer.
        :param data: 2‑D array of shape (kernel_size, kernel_size) for the quanvolution step.
        :return: numpy array produced by the quantum fully‑connected layer.
        """
        conv_out = self.conv.run(data)
        return self.fc.run(conv_out)

__all__ = ["HybridLayer"]
