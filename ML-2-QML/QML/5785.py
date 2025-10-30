import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
from typing import Sequence

class QuantumConvFilter:
    """Quantum circuit implementing a convolutional filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 shots: int = 1024, backend: AerSimulator | None = None) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or AerSimulator()
        self.circuit = QuantumCircuit(self.n_qubits)
        self.params = [Parameter(f'theta_{i}') for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.params[i], i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        data = data.reshape(1, self.n_qubits)
        param_binds = []
        for row in data:
            bind = {p: np.pi if val > self.threshold else 0.0 for p, val in zip(self.params, row)}
            param_binds.append(bind)
        job = execute(self.circuit, backend=self.backend,
                      shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = sum(sum(int(bit) for bit in key) * val for key, val in result.items())
        return counts / (self.shots * self.n_qubits)

class QuantumKernel:
    """Variational quantum kernel using a rotation‑only Ansatz."""
    def __init__(self, n_wires: int = 4, shots: int = 1024,
                 backend: AerSimulator | None = None) -> None:
        self.n_wires = n_wires
        self.shots = shots
        self.backend = backend or AerSimulator()
        self.circuit = QuantumCircuit(n_wires)
        self.params = [Parameter(f'theta_{i}') for i in range(n_wires)]
        for i in range(n_wires):
            self.circuit.ry(self.params[i], i)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, x: np.ndarray, y: np.ndarray) -> float:
        job_x = execute(self.circuit.bind_parameters({p: val for p, val in zip(self.params, x)}),
                        backend=self.backend, shots=self.shots)
        job_y = execute(self.circuit.bind_parameters({p: val for p, val in zip(self.params, y)}),
                        backend=self.backend, shots=self.shots)
        res_x = job_x.result().get_counts(self.circuit)
        res_y = job_y.result().get_counts(self.circuit)
        p_x0 = res_x.get('0'*self.n_wires, 0) / self.shots
        p_y0 = res_y.get('0'*self.n_wires, 0) / self.shots
        return np.sqrt(p_x0 * p_y0)

def kernel_matrix(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel.run(x, y) for y in b] for x in a])

class ConvGen175:
    """Quantum‑enhanced convolution+kernel combined module."""
    def __init__(self,
                 kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 conv_shots: int = 512,
                 kernel_shots: int = 512) -> None:
        self.conv_filter = QuantumConvFilter(kernel_size=kernel_size,
                                            threshold=conv_threshold,
                                            shots=conv_shots)
        self.kernel = QuantumKernel(n_wires=4, shots=kernel_shots)

    def run_conv(self, data: np.ndarray) -> float:
        return self.conv_filter.run(data)

    def run_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.kernel.run(x, y)

    def kernel_matrix(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        return kernel_matrix(a, b)

__all__ = ["ConvGen175", "QuantumConvFilter", "QuantumKernel", "kernel_matrix"]
