"""Quantum implementation of a fully connected layer using parameterized circuits."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
from typing import Iterable

class FullyConnectedLayer:
    """Parameterized quantum circuit that emulates a fully connected layer."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.theta = Parameter("θ")
        self.circuit = QuantumCircuit(n_qubits)
        # Entangling layer
        self.circuit.h(range(n_qubits))
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.barrier()
        # Parameterized rotation
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        exp_values = []
        for theta in thetas:
            bound_circ = self.circuit.bind_parameters({self.theta: theta})
            job = execute(bound_circ, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(bound_circ)
            exp = 0.0
            for state, cnt in counts.items():
                prob = cnt / self.shots
                for bit in state[::-1]:  # little endian
                    exp += (1 if bit == '0' else -1) * prob
            exp_values.append(exp)
        return np.array(exp_values)

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        shift = np.pi / 2
        grads = []
        for theta in thetas:
            exp_pos = self.run([theta + shift])[0]
            exp_neg = self.run([theta - shift])[0]
            grads.append((exp_pos - exp_neg) / 2)
        return np.array(grads)

def FCL(n_qubits: int, shots: int = 1024) -> FullyConnectedLayer:
    """Return a FullyConnectedLayer instance configured for the given qubits."""
    return FullyConnectedLayer(n_qubits, shots=shots)

__all__ = ["FullyConnectedLayer", "FCL"]
