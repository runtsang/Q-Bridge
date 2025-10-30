"""HybridSelfAttentionClassifier – quantum implementation."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator

class QuantumSelfAttention:
    """Parametrised self‑attention circuit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend: qiskit.providers.Provider,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        # Normalise to a probability vector over all bitstrings
        probs = np.array(
            [counts.get(bin(i)[2:].zfill(self.n_qubits), 0) for i in range(2**self.n_qubits)],
            dtype=float,
        )
        probs /= shots
        return probs


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> tuple[QuantumCircuit, list, list, list[SparsePauliOp]]:
    """Variational classifier ansatz with explicit encoding."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.rx(encoding[qubit], qubit)
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class QuantumClassifierCircuit:
    """Expectation‑value head built from the variational ansatz."""
    def __init__(self, num_qubits: int, depth: int, backend, shots: int):
        self.circuit, self.encoding_params, self.weight_params, self.observables = (
            build_classifier_circuit(num_qubits, depth)
        )
        self.backend = backend
        self.shots = shots

    def run(self, inputs: np.ndarray) -> np.ndarray:
        # Bind encoding parameters to the input vector
        param_bindings = {p: v for p, v in zip(self.encoding_params, inputs)}
        bound_circ = self.circuit.bind_parameters(param_bindings)
        compiled = transpile(bound_circ, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result().get_counts(bound_circ)
        expectations = []
        for obs in self.observables:
            exp = 0.0
            for state, count in result.items():
                prob = count / self.shots
                bitstring = state[::-1]  # reverse to match qubit order
                val = 1 if bitstring[obs.qubits[0]] == "1" else -1
                exp += val * prob
            expectations.append(exp)
        return np.array(expectations)


class HybridSelfAttentionClassifier:
    """Hybrid self‑attention followed by a quantum classifier head."""
    def __init__(
        self,
        n_qubits: int = 4,
        depth: int = 2,
        backend=None,
        shots: int = 1024,
    ):
        if backend is None:
            backend = AerSimulator()
        self.self_attention = QuantumSelfAttention(n_qubits)
        self.classifier = QuantumClassifierCircuit(
            n_qubits, depth, backend, shots
        )
        self.backend = backend
        self.shots = shots

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # Execute self‑attention circuit
        attn_probs = self.self_attention.run(
            backend=self.backend,
            rotation_params=rotation_params,
            entangle_params=entangle_params,
            shots=self.shots,
        )
        # Feed the probability vector into the classifier
        logits = self.classifier.run(attn_probs)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs


__all__ = ["HybridSelfAttentionClassifier"]
