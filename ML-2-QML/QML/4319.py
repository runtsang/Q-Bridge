"""Quantum implementations for fraud detection: a quantum kernel and a parameterized sampler QNN."""
import numpy as np
import torch
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

class QuantumKernel:
    """Quantum kernel that encodes inputs as Ry‑rotated states and evaluates overlap."""
    def __init__(self, n_qubits: int = 2):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend('statevector_simulator')

    def _encode(self, x: np.ndarray) -> QuantumCircuit:
        """Encode a 2‑dim vector into a 2‑qubit state via Ry rotations."""
        qc = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(x):
            qc.ry(val, i)
        return qc

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute |<psi(x)|psi(y)>|^2."""
        qc_x = self._encode(x)
        qc_y = self._encode(y)
        sv_x = execute(qc_x, self.backend).result().get_statevector()
        sv_y = execute(qc_y, self.backend).result().get_statevector()
        return abs(np.vdot(sv_x, sv_y))**2

    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute Gram matrix between two sets of inputs."""
        return np.array([[self.evaluate(x, y) for y in b] for x in a])

class SamplerQNN:
    """Parameterised quantum sampler QNN for fraud detection."""
    def __init__(self):
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)
        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        for i in range(4):
            self.circuit.ry(self.weight_params[i], i % 2)
        self.circuit.cx(0, 1)
        self.backend = Aer.get_backend('qasm_simulator')
        self.shots = 1024

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """Return a probability distribution over two outcomes for each input."""
        probs = []
        for inp in inputs:
            param_bind = {
                self.input_params[0]: inp[0],
                self.input_params[1]: inp[1]
            }
            # Bind weight parameters to zero for demonstration; in practice they are trainable.
            param_bind.update({self.weight_params[i]: 0.0 for i in range(4)})
            job = execute(
                self.circuit.bind_parameters(param_bind),
                self.backend,
                shots=self.shots
            )
            result = job.result().get_counts()
            prob0 = result.get('00', 0) / self.shots
            prob1 = result.get('01', 0) / self.shots
            probs.append([prob0, prob1])
        return np.array(probs)

__all__ = ["QuantumKernel", "SamplerQNN"]
