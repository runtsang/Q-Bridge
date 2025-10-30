python
"""Parameterized variational circuit mimicking a fully‑connected layer.

The circuit consists of a chain of single‑qubit Ry rotations followed
by a linear CNOT entanglement layer.  The parameters are applied as
Ry gates on every qubit, and the expectation value of Pauli‑Z on
qubit 0 is returned as the layer output.  A simple parameter‑shift
gradient routine and a lightweight training helper are provided.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

class FCLayer:
    """Variational circuit with depth‑controlled entanglement."""
    def __init__(self, n_qubits: int = 1, depth: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.params = ParameterVector("θ", length=n_qubits * depth)
        self._build_circuit()

    def _build_circuit(self):
        self.circuit = QuantumCircuit(self.n_qubits)
        idx = 0
        for _ in range(self.depth):
            for q in range(self.n_qubits):
                self.circuit.ry(self.params[idx], q)
                idx += 1
            # Linear CNOT chain
            for q in range(self.n_qubits - 1):
                self.circuit.cx(q, q + 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit and return the Z‑expectation on qubit 0."""
        bound = {self.params[i]: float(thetas[i]) for i in range(len(thetas))}
        bound_circ = self.circuit.bind_parameters(bound)
        job = execute(bound_circ, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circ)
        exp = 0.0
        for state, count in counts.items():
            prob = count / self.shots
            bit = int(state[-1])  # qubit 0 is the least‑significant bit
            exp += (1 if bit == 0 else -1) * prob
        return np.array([exp])

    # ------------------------------------------------------------------
    # Gradient via parameter‑shift rule
    # ------------------------------------------------------------------
    def gradient(self, thetas: np.ndarray) -> np.ndarray:
        """Return the gradient of the expectation w.r.t. each parameter."""
        grads = np.zeros_like(thetas, dtype=np.float32)
        shift = np.pi / 2  # parameter‑shift step
        for i in range(len(thetas)):
            plus = thetas.copy()
            minus = thetas.copy()
            plus[i] += shift
            minus[i] -= shift
            f_plus = self.run(plus)[0]
            f_minus = self.run(minus)[0]
            grads[i] = (f_plus - f_minus) * 0.5
        return grads

    # ------------------------------------------------------------------
    # One‑step training helper
    # ------------------------------------------------------------------
    def train_one_step(
        self,
        thetas: np.ndarray,
        target: float,
        lr: float = 1e-2,
    ) -> np.ndarray:
        """Perform a single gradient‑descent update towards the target."""
        y = self.run(thetas)[0]
        grads = self.gradient(thetas)
        new_thetas = thetas - lr * 2 * (y - target) * grads
        return new_thetas

__all__ = ["FCLayer"]
