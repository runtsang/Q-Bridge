"""Quantum fully connected layer with parameter shift gradients and multi‑qubit support."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from typing import Iterable

class FullyConnectedLayer:
    """Parameterized quantum circuit simulating a fully connected layer."""
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend('qasm_simulator')
        self.theta = Parameter("theta")
        self._build_circuit()

    def _build_circuit(self):
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.h(range(self.n_qubits))
        self.circuit.barrier()
        # Parameterized rotation on each qubit
        for q in range(self.n_qubits):
            self.circuit.ry(self.theta, q)
        # Simple entanglement layer
        for q in range(self.n_qubits - 1):
            self.circuit.cx(q, q+1)
        self.circuit.barrier()
        # Measurement
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit for each theta and return expectation of Z."""
        results = []
        for theta in thetas:
            bound = self.circuit.bind_parameters({self.theta: theta})
            job = execute(bound, self.backend, shots=self.shots)
            counts = job.result().get_counts(bound)
            # Convert counts to expectation value of PauliZ on all qubits
            exp = 0.0
            for bitstring, count in counts.items():
                prob = count / self.shots
                # Map '0'->+1, '1'->-1 for each qubit and multiply
                z = 1
                for bit in bitstring[::-1]:  # Qiskit orders qubits reversed
                    z *= 1 if bit == '0' else -1
                exp += z * prob
            results.append(exp)
        return np.array(results).reshape(-1, 1)

    def parameter_shift_gradient(self, thetas: Iterable[float], shift: float = np.pi/2) -> np.ndarray:
        """Compute gradient using parameter shift rule."""
        grads = []
        for theta in thetas:
            theta_plus = theta + shift
            theta_minus = theta - shift
            f_plus = self.run([theta_plus])[0,0]
            f_minus = self.run([theta_minus])[0,0]
            grad = 0.5 * (f_plus - f_minus)
            grads.append(grad)
        return np.array(grads).reshape(-1, 1)

def FCL(n_qubits: int = 1, shots: int = 1024):
    """Convenience factory mirroring the classical API."""
    return FullyConnectedLayer(n_qubits=n_qubits, shots=shots)
