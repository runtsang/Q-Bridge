from qiskit import QuantumCircuit, Aer, execute
import numpy as np

class HybridSampler:
    """
    Quantum hybrid sampler mirroring the classical implementation.
    A classical linear mapping generates rotation angles for a parametric
    2‑qubit circuit. The circuit is executed on a Qiskit QASM simulator
    to obtain measurement probabilities.
    """

    def __init__(self, weight_dim: int = 4, seed: int | None = None) -> None:
        # Classical weight generation matrix (2 → weight_dim)
        self.W = np.random.randn(2, weight_dim) * 0.1
        self.b = np.zeros(weight_dim)
        if seed is not None:
            np.random.seed(seed)

    def _build_circuit(self, inputs: np.ndarray, weights: np.ndarray) -> QuantumCircuit:
        """
        Build a 2‑qubit circuit with input and weight rotations.
        """
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        qc.measure_all()
        return qc

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the hybrid sampler on a single 2‑dim input vector.
        Returns a 2‑dim probability vector: (prob qubit0=0, prob qubit0=1).
        """
        # Generate weight angles from classical linear mapping
        weights = self.W @ inputs + self.b
        qc = self._build_circuit(inputs, weights)

        backend = Aer.get_backend("qasm_simulator")
        job = execute(qc, backend=backend, shots=1024)
        result = job.result()
        counts = result.get_counts(qc)

        probs = np.zeros(2)
        for bitstring, count in counts.items():
            if bitstring[0] == '0':
                probs[0] += count
            else:
                probs[1] += count
        probs /= 1024
        return probs

__all__ = ["HybridSampler"]
