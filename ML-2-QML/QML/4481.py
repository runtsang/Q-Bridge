import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from Conv import Conv
from QuantumClassifierModel import build_classifier_circuit

class SelfAttentionGen162:
    """Quantum‑only self‑attention module that uses a swap‑test to
    compute similarity between two classical vectors and then feeds
    the weighted result into a variational classifier."""
    def __init__(self,
                 n_qubits: int = 4,
                 classifier_depth: int = 2):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")
        self.conv = Conv()
        self.classifier_circuit, self.enc_params, self.weight_params, self.observables = build_classifier_circuit(n_qubits, classifier_depth)

    def _swap_test(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Return an estimate of the inner product via a swap‑test."""
        circuit = QuantumCircuit(self.n_qubits + 1, 1)
        circuit.h(0)
        for i in range(self.n_qubits):
            circuit.rx(vec1[i], i + 1)
            circuit.rx(vec2[i], i + 1)
        circuit.cz(0, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        job = qiskit.execute(circuit, self.backend, shots=1024)
        counts = job.result().get_counts(circuit)
        prob0 = counts.get("0", 0) / 1024
        return 2 * prob0 - 1  # similarity in [-1,1]

    def run_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Weighted value vector using quantum similarity."""
        similarity = self._swap_test(query, key)
        return similarity * value

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Full forward pass:
        1) Use the quantum filter to extract a scalar feature.
        2) Build query/key/value vectors from that scalar.
        3) Compute the attention weight quantum‑ly.
        4) Feed the weighted vector into the variational classifier and
           return the expectation values of the Pauli‑Z observables.
        """
        # 1. Quantum filter
        feature = self.conv.run(data)  # scalar

        # 2. Build vectors
        vec = np.full(self.n_qubits, feature)
        attn_vec = self.run_attention(vec, vec, vec)

        # 3. Bind the attention vector to the classifier circuit
        bind = {p: float(v) for p, v in zip(self.enc_params, attn_vec)}
        circuit = self.classifier_circuit.bind_parameters(bind)
        circuit.measure_all()

        # 4. Execute and compute expectations
        job = qiskit.execute(circuit, self.backend, shots=1024)
        result = job.result().get_counts(circuit)
        expectations = np.zeros(self.n_qubits)
        for i in range(self.n_qubits):
            zero = sum(count for bitstring, count in result.items()
                       if bitstring[self.n_qubits-1-i] == "0")
            one  = sum(count for bitstring, count in result.items()
                       if bitstring[self.n_qubits-1-i] == "1")
            expectations[i] = (zero - one) / 1024
        return expectations

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.run(data)

__all__ = ["SelfAttentionGen162"]
