import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.providers import Backend
from typing import Iterable, List

class FCL:
    """
    Parameterized quantum circuit emulating a fully connected layer.
    Supports full or linear entanglement, parameter‑shift gradient
    computation, and gradient‑descent training on a qasm simulator.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        backend: Backend | None = None,
        shots: int = 1024,
        entanglement: str = "full",
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.entanglement = entanglement
        self.theta = ParameterVector("θ", self.n_qubits)
        self._build_circuit()

    def _build_circuit(self) -> None:
        qc = QuantumCircuit(self.n_qubits)
        # Initial Hadamard layer
        qc.h(range(self.n_qubits))
        # Parameterized Ry rotations
        for i in range(self.n_qubits):
            qc.ry(self.theta[i], i)
        # Entanglement
        if self.entanglement == "full":
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qc.cx(i, j)
        elif self.entanglement == "linear":
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        else:
            raise ValueError(f"Unsupported entanglement: {self.entanglement}")
        # Measurement of Pauli‑Z expectation
        self.circuit = qc
        self.transpiled = transpile(self.circuit, self.backend)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for a list of parameters and return the
        expectation value of Pauli‑Z on each qubit as a NumPy array.
        """
        bound_circuit = self.transpiled.bind_parameters(
            {self.theta[i]: thetas[i] for i in range(self.n_qubits)}
        )
        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)
        probs = np.array([counts.get(f"{i:0{self.n_qubits}b}", 0) for i in range(2**self.n_qubits)])
        probs = probs / self.shots
        exp_z = np.zeros(self.n_qubits)
        for bitstring, p in zip(counts.keys(), probs):
            bits = np.array([int(b) for b in bitstring])
            z_vals = 1 - 2 * bits  # 0 -> +1, 1 -> -1
            exp_z += z_vals * p
        return exp_z

    def parameter_shift_gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the expectation vector w.r.t. the parameters
        using the parameter‑shift rule.
        """
        shift = np.pi / 2
        grads = np.zeros_like(thetas, dtype=float)
        for i in range(len(thetas)):
            shifted_plus = list(thetas)
            shifted_minus = list(thetas)
            shifted_plus[i] += shift
            shifted_minus[i] -= shift
            exp_plus = self.run(shifted_plus)
            exp_minus = self.run(shifted_minus)
            grads[i] = (exp_plus - exp_minus) / 2.0
        return grads

    def train(
        self,
        thetas: List[float],
        target: np.ndarray,
        lr: float = 0.01,
        epochs: int = 100,
        verbose: bool = True,
    ) -> List[float]:
        """
        Gradient‑descent training of the circuit parameters to match a target
        expectation vector. Returns the optimized parameters.
        """
        params = np.array(thetas, dtype=float)
        for epoch in range(epochs):
            preds = self.run(params)
            loss = np.mean((preds - target) ** 2)
            grads = self.parameter_shift_gradient(params)
            params -= lr * grads
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.6f}")
        return params.tolist()

    def save_params(self, path: str) -> None:
        """Persist the circuit parameters to disk."""
        np.save(path, np.array(self.theta))

    def load_params(self, path: str) -> None:
        """Load circuit parameters from disk."""
        self.theta = np.load(path).tolist()

__all__ = ["FCL"]
