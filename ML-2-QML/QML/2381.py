"""Hybrid quanvolution + variational classifier for quantum experiments.

The quantum implementation mirrors the classical architecture:
  * A small quanvolution sub‑circuit that encodes a ``kernel_size``×``kernel_size``
    image patch into RX rotations, applies a random two‑layer ansatz, and measures
    all qubits.  The average probability of observing |1⟩ is used as a
    feature.
  * A variational classifier that operates on ``num_features`` qubits.  The first
    qubit is fed the result of the quanvolution; the remaining qubits encode
    the remaining features.  The circuit consists of alternating layers of
    single‑qubit rotations and nearest‑neighbour CZ gates, followed by a set
    of Z‑observables.  The expectation values are averaged to produce a
    probability for class 0; the complementary probability is 1‑p.

The ``run`` method accepts a tuple ``(patch, features)`` where
  * ``patch`` is a 2‑D array of shape ``(kernel_size, kernel_size)``.
  * ``features`` is a 1‑D array of length ``num_features‑1``.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Iterable

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class HybridConvClassifier:
    """Quantum analogue of the classical HybridConvClassifier."""

    def __init__(
        self,
        kernel_size: int = 2,
        depth: int = 2,
        num_features: int = 10,
        threshold: float = 0.5,
        backend=None,
        shots: int = 1024,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square quantum filter.
        depth : int
            Depth of the variational classifier ansatz.
        num_features : int
            Total number of qubits used in the classifier part.
            The first qubit receives the quanvolution output.
        threshold : float
            Classical threshold for encoding the patch.
        backend : qiskit backend or None
            If None, the Aer qasm simulator is used.
        shots : int
            Number of shots for each execution.
        """
        self.kernel_size = kernel_size
        self.depth = depth
        self.num_features = num_features
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Build quanvolution sub‑circuit
        self.conv_circuit = self._build_quanvolution()

        # Build variational classifier
        self.classifier_circuit, self.enc_params, self.var_params, self.observables = (
            self._build_classifier()
        )

    def _build_quanvolution(self) -> QuantumCircuit:
        """Construct a small quanvolution circuit for a patch."""
        n_qubits = self.kernel_size ** 2
        qc = QuantumCircuit(n_qubits)
        theta = ParameterVector("theta", n_qubits)
        # Encode the patch via RX rotations
        for i in range(n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        # Add a simple random two‑layer ansatz
        for _ in range(2):
            for i in range(n_qubits):
                qc.rz(np.random.rand() * 2 * np.pi, i)
            for i in range(n_qubits - 1):
                qc.cz(i, i + 1)
        qc.measure_all()
        return qc

    def _build_classifier(
        self,
    ) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], list[SparsePauliOp]]:
        """Construct a layered variational classifier."""
        qc = QuantumCircuit(self.num_features)
        encoding = ParameterVector("x", self.num_features)
        weights = ParameterVector("theta", self.num_features * self.depth)

        # Data encoding
        for param, qubit in zip(encoding, range(self.num_features)):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_features):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_features - 1):
                qc.cz(qubit, qubit + 1)

        # Observables: Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_features - i - 1))
            for i in range(self.num_features)
        ]
        return qc, list(encoding), list(weights), observables

    def _quanvolution_run(self, patch: np.ndarray) -> float:
        """Execute the quanvolution circuit on a single patch."""
        # Flatten and bind parameters
        flat = patch.reshape(1, -1)
        param_binds = []
        for row in flat:
            bind = {self.conv_circuit.parameters[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(row)}
            param_binds.append(bind)

        job = execute(
            self.conv_circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.conv_circuit)

        # Compute average probability of measuring |1> across all qubits
        total_ones = 0
        total_counts = 0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * cnt
            total_counts += cnt
        avg_prob = total_ones / (total_counts * self.kernel_size ** 2)
        return avg_prob

    def run(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float]:
        """
        Execute the full hybrid circuit.

        Parameters
        ----------
        data : Tuple[np.ndarray, np.ndarray]
            ``(patch, features)`` where
            * ``patch`` has shape ``(kernel_size, kernel_size)``.
            * ``features`` has length ``num_features - 1`` and
              contains the remaining feature values.

        Returns
        -------
        Tuple[float, float]
            Probabilities for classes 0 and 1.
        """
        patch, features = data
        if patch.ndim!= 2 or patch.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError("Patch must be of shape (kernel_size, kernel_size).")
        if features.ndim!= 1 or features.size!= self.num_features - 1:
            raise ValueError(f"Features must be 1‑D of length {self.num_features - 1}.")

        # Quanvolution feature
        conv_feature = self._quanvolution_run(patch)

        # Prepare parameter bindings for classifier
        # First qubit receives conv_feature, remaining qubits receive features
        param_bind = {}
        for i, val in enumerate([conv_feature] + features.tolist()):
            param_bind[self.classifier_circuit.parameters[i]] = val

        job = execute(
            self.classifier_circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.classifier_circuit)

        # Compute expectation values of Z on each qubit
        exp_vals = []
        for i in range(self.num_features):
            exp = 0
            for bitstring, cnt in counts.items():
                bit = int(bitstring[self.num_features - 1 - i])  # reverse order
                exp += (1 if bit == 1 else -1) * cnt
            exp /= sum(counts.values())
            exp_vals.append(exp)

        # Average the expectation values to obtain a single score
        score = sum(exp_vals) / self.num_features
        prob0 = (1 + score) / 2
        prob1 = 1 - prob0
        return prob0, prob1


__all__ = ["HybridConvClassifier"]
