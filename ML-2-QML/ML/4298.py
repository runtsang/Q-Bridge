import math
import numpy as np
import torch
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

class SelfAttention:
    """
    Hybrid self‑attention layer that blends a classical torch attention backbone
    with an optional quantum expectation head.  The classical part follows the
    standard scaled dot‑product attention; the quantum part replaces the
    soft‑max weight calculation with a variational circuit that estimates
    the attention distribution from the query‑key similarity.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    n_qubits : int, optional
        Number of qubits used in the quantum head (default 4).
    backend : qiskit.providers.Backend, optional
        Quantum backend; if None an Aer simulator is used.
    shots : int, optional
        Number of shots for the quantum execution (default 1024).
    """

    def __init__(self, embed_dim: int, n_qubits: int = 4,
                 backend=None, shots: int = 1024):
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._qc_template = self._build_qc_template()

    def _build_qc_template(self):
        """Template for the variational circuit used to compute a single weight."""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        theta = ParameterVector("theta", length=self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(theta[i], i)
        for i in range(self.n_qubits):
            qc.cx(i, (i + 1) % self.n_qubits)
        qc.measure_all()
        return qc

    def _classical_attention(self, queries, keys, values):
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, values)

    def _quantum_weight(self, query, key):
        """Run the variational circuit for a single query‑key dot product."""
        angle = float(np.dot(query, key)) * math.pi
        param_dict = {f"theta_{i}": angle for i in range(self.n_qubits)}
        qc = self._qc_template.bind_parameters(param_dict)
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        prob_zero = counts.get("0" * self.n_qubits, 0) / self.shots
        return torch.tensor(prob_zero, dtype=torch.float32)

    def run(self, inputs: np.ndarray,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            use_quantum: bool = False) -> np.ndarray:
        """
        Forward pass.  If *use_quantum* is True the attention weights are
        derived from the quantum head; otherwise a purely classical soft‑max
        is applied.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, seq_len, embed_dim).
        rotation_params, entangle_params : np.ndarray
            Parameters for the quantum circuit (not used in the classical
            branch but required for API compatibility).
        use_quantum : bool
            Flag to enable the quantum head.
        """
        batch, seq_len, _ = inputs.shape
        queries = torch.tensor(inputs, dtype=torch.float32)
        keys = queries.clone()
        values = queries.clone()
        if use_quantum:
            weights = torch.zeros((batch, seq_len, seq_len), dtype=torch.float32)
            for b in range(batch):
                for i in range(seq_len):
                    for j in range(seq_len):
                        w = self._quantum_weight(queries[b, i], keys[b, j])
                        weights[b, i, j] = w
            weights = weights / weights.sum(dim=-1, keepdim=True)
        else:
            weights = torch.softmax(torch.matmul(queries, keys.transpose(-2, -1)) /
                                   math.sqrt(self.embed_dim), dim=-1)
        output = torch.matmul(weights, values)
        return output.numpy()

__all__ = ["SelfAttention"]
