import numpy as np
import qiskit
from qiskit import QuantumCircuit
from typing import Iterable

class HybridFCLAttention:
    """
    Quantum hybrid fully‑connected + self‑attention layer.

    Mirrors the classical interface but implements the attention block
    with a parameterised quantum circuit.  Parameters are supplied as a
    flat iterable ``thetas`` containing:

    * ``n_features * embed_dim`` weights for a classical linear map
    * ``embed_dim`` biases for the linear map
    * ``embed_dim * 3`` rotation angles for the quantum circuit
    * ``(embed_dim - 1) * embed_dim`` entanglement angles

    The linear part is evaluated classically; the attention part is
    executed on a qasm simulator.  This design enables hybrid training
    where the quantum circuit can be differentiated via parameter‑shift
    or gradient‑free methods.
    """
    def __init__(self, n_features: int, embed_dim: int = 4,
                 backend=None, shots: int = 1024) -> None:
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

    def _build_circuit(self, rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.embed_dim, self.embed_dim)
        for i in range(self.embed_dim):
            qc.rx(rotation_params[i, 0], i)
            qc.ry(rotation_params[i, 1], i)
            qc.rz(rotation_params[i, 2], i)
        for i in range(self.embed_dim - 1):
            qc.crx(entangle_params[i, i], i, i + 1)
        qc.measure(range(self.embed_dim), range(self.embed_dim))
        return qc

    def run(self, thetas: Iterable[float], inputs: np.ndarray) -> np.ndarray:
        # Parse the theta vector
        thetas = np.asarray(thetas, dtype=np.float32)
        w_len = self.n_features * self.embed_dim
        weight = thetas[:w_len].reshape(self.embed_dim, self.n_features)
        bias = thetas[w_len:w_len + self.embed_dim]

        start = w_len + self.embed_dim
        rotation_len = self.embed_dim * 3
        rotation_params = thetas[start:start + rotation_len].reshape(self.embed_dim, 3)
        entangle_len = (self.embed_dim - 1) * self.embed_dim
        entangle_params = thetas[start + rotation_len:start + rotation_len + entangle_len].reshape(self.embed_dim - 1, self.embed_dim)

        # Classical linear map
        linear_out = inputs @ weight.T + bias

        # Quantum attention
        qc = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(qc, self.backend, shots=self.shots)
        result = job.result().get_counts(qc)

        # Compute expectation of Z on each qubit
        expectations = np.zeros(self.embed_dim)
        for state, count in result.items():
            prob = count / self.shots
            for i, bit in enumerate(reversed(state)):
                expectations[i] += (1 if bit == '0' else -1) * prob
        return expectations.reshape(1, -1)

def FCL(n_features: int = 1, embed_dim: int = 4,
        backend=None, shots: int = 1024) -> HybridFCLAttention:
    """
    Factory that returns a quantum HybridFCLAttention instance.
    """
    return HybridFCLAttention(n_features, embed_dim, backend, shots)

__all__ = ["HybridFCLAttention", "FCL"]
