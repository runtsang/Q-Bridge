"""Hybrid quantum convolution + classifier.

The quantum implementation mirrors the classical structure:
a quanvolution filter is applied to the input data, its measurement
statistics are used as a classical encoding for a variational ansatz,
and the expectation values of Z observables provide the logits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class HybridQuanvClassifier:
    """
    Quantum analogue of HybridConvClassifier.  It first applies a
    quanvolution filter to the input, then uses the resulting
    measurement statistics as a classical encoding for a variational
    circuit.  The expectation values of Z observables are returned as
    logits, matching the shape of the classical classifier.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 127,
                 classifier_depth: int = 2,
                 shots: int = 1024,
                 backend: qiskit.providers.Backend | None = None) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

        # Quanvolution filter ------------------------------------------------
        self._filter_circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._filter_circuit.rx(self.theta[i], i)
        self._filter_circuit.barrier()
        self._filter_circuit += random_circuit(self.n_qubits, 2)
        self._filter_circuit.measure_all()

        # Classifier ansatz ---------------------------------------------------
        self.classifier_circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits=self.n_qubits, depth=classifier_depth
        )

    # -------------------------------------------------------------------------
    def _run_filter(self, data: np.ndarray) -> float:
        """Execute the quanvolution filter and return the mean |1> probability."""
        flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in flat:
            bind = {theta: np.pi if val > self.threshold else 0 for theta, val in zip(self.theta, dat)}
            param_binds.append(bind)

        job = qiskit.execute(
            self._filter_circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._filter_circuit)

        # Compute average number of |1> over all shots
        total = self.shots * self.n_qubits
        count = 0
        for bitstring, freq in result.items():
            count += sum(int(b) for b in bitstring) * freq
        return count / total

    # -------------------------------------------------------------------------
    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the full hybrid circuit and return logits as a numpy array.
        The input should be a 2‑D array of shape (kernel_size, kernel_size).
        """
        # 1. Run the quanvolution filter
        prob = self._run_filter(data)

        # 2. Build encoding: a simple thresholding of the probability
        encoding_vals = [np.pi if prob > 0.5 else 0] * self.n_qubits
        enc_bind = dict(zip(self.encoding, encoding_vals))

        # 3. Zero‑initialize all variational weights
        weight_bind = {w: 0 for w in self.weights}

        # 4. Execute the classifier circuit
        param_binds = [ {**enc_bind, **weight_bind} ]
        job = qiskit.execute(
            self.classifier_circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.classifier_circuit)

        # 5. Compute expectation values of Z observables
        logits = np.zeros(self.n_qubits, dtype=float)
        total_counts = self.shots
        for bitstring, freq in result.items():
            for i in range(self.n_qubits):
                # Qiskit uses little‑endian ordering; bitstring[0] is qubit 0
                bit = int(bitstring[::-1][i])  # reverse for little‑endian
                logits[i] += (1 if bit == 0 else -1) * freq

        logits /= total_counts
        return logits

def Conv() -> HybridQuanvClassifier:
    """
    Factory that returns a ready‑to‑use HybridQuanvClassifier.
    Mirrors the original Conv() API so existing code continues to work.
    """
    return HybridQuanvClassifier()

__all__ = ["HybridQuanvClassifier", "Conv"]
