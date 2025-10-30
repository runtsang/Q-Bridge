"""Quantum hybrid QCNN + autoencoder with graph adjacency.

This module builds a variational circuit that mirrors the classical
QCNNHybridAutoencoder.  It uses a ZFeatureMap, a convolutional ansatz
with pooling, and an autoencoder sub‑circuit based on RealAmplitudes
and a swap‑test.  The resulting EstimatorQNN is wrapped in a
FastBaseEstimator for rapid expectation evaluation.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.primitives import Estimator as Estimator
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Pauli
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

from typing import Iterable, List, Sequence

# ---------------------------------------------------------------------------

class QCNNHybridAutoencoder:
    """
    Variational circuit that combines a QCNN ansatz with a quantum autoencoder.
    """

    def __init__(self, input_dim: int, latent_dim: int = 3, trash_dim: int = 2) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim

        algorithm_globals.random_seed = 42
        self.estimator = Estimator()

        # Feature map
        self.feature_map = ZFeatureMap(input_dim)

        # Convolutional ansatz
        self.ansatz = self._build_ansatz()

        # Autoencoder sub‑circuit
        self.auto_circuit = self._build_autoencoder()

        # Combine
        self.circuit = QuantumCircuit(input_dim)
        self.circuit.compose(self.feature_map, range(input_dim), inplace=True)
        self.circuit.compose(self.ansatz, range(input_dim), inplace=True)
        self.circuit.compose(self.auto_circuit, range(input_dim), inplace=True)

        # Observable: measure Z on first qubit (for classification)
        self.observable = Pauli('Z' + 'I' * (input_dim - 1))

        # QNN wrapper
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=[self.observable],
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters + self.auto_circuit.parameters,
            estimator=self.estimator,
        )

    # -----------------------------------------------------------------------
    # Convolutional ansatz
    def _build_conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = self._conv_unitary(params[idx : idx + 3], q1, q2)
            qc.append(sub, [q1, q2])
            idx += 3
        return qc

    def _conv_unitary(self, params, q1, q2):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, q2)
        sub.cx(q2, q1)
        sub.rz(params[0], q1)
        sub.ry(params[1], q2)
        sub.cx(q1, q2)
        sub.ry(params[2], q2)
        sub.cx(q2, q1)
        sub.rz(np.pi / 2, q1)
        return sub

    def _build_pool_layer(self, sources, sinks, prefix: str) -> QuantumCircuit:
        num = len(sources) + len(sinks)
        qc = QuantumCircuit(num)
        params = ParameterVector(prefix, length=len(sources) * 3)
        idx = 0
        for s, t in zip(sources, sinks):
            sub = self._pool_unitary(params[idx : idx + 3], s, t)
            qc.append(sub, [s, t])
            idx += 3
        return qc

    def _pool_unitary(self, params, s, t):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, t)
        sub.cx(t, s)
        sub.rz(params[0], s)
        sub.ry(params[1], t)
        sub.cx(s, t)
        sub.ry(params[2], t)
        return sub

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.input_dim)
        # First conv + pool
        qc.append(self._build_conv_layer(self.input_dim, "c1"), range(self.input_dim))
        qc.append(self._build_pool_layer(range(self.input_dim // 2), range(self.input_dim // 2, self.input_dim), "p1"), range(self.input_dim))
        # Second conv + pool
        qc.append(self._build_conv_layer(self.input_dim // 2, "c2"), range(self.input_dim // 2))
        qc.append(self._build_pool_layer(range(self.input_dim // 4), range(self.input_dim // 4, self.input_dim // 2), "p2"), range(self.input_dim // 2))
        # Third conv + pool
        qc.append(self._build_conv_layer(self.input_dim // 4, "c3"), range(self.input_dim // 4))
        qc.append(self._build_pool_layer([0], [1], "p3"), range(self.input_dim // 4))
        return qc

    # -----------------------------------------------------------------------
    # Autoencoder sub‑circuit
    def _build_autoencoder(self) -> QuantumCircuit:
        num = self.latent_dim + 2 * self.trash_dim + 1
        qc = QuantumCircuit(num)
        # Ansatz
        ans = RealAmplitudes(num - 1, reps=5)
        qc.append(ans, range(num - 1))
        qc.barrier()
        # Swap test
        aux = num - 1
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)
        # No classical register needed for the estimator; measurement is handled by EstimatorQNN
        return qc

    # -----------------------------------------------------------------------
    # Fast estimator wrapper
    def evaluate(
        self,
        inputs: Sequence[np.ndarray],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each input.  If shots is provided,
        returns noisy estimates.
        """
        param_sets = [list(inp) for inp in inputs]
        results = self.qnn.evaluate(param_sets)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy_row = [rng.normal(complex(val.real, val.imag), 1 / shots) for val in row]
            noisy.append(noisy_row)
        return noisy

    # -----------------------------------------------------------------------
    # Fidelity adjacency from measurement states
    def fidelity_adjacency(
        self,
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Build a weighted adjacency matrix from state fidelities.
        """
        n = len(states)
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                fid = abs(states[i].data.conj().T @ states[j].data) ** 2
                if fid >= threshold:
                    matrix[i, j] = matrix[j, i] = 1.0
                elif secondary is not None and fid >= secondary:
                    matrix[i, j] = matrix[j, i] = secondary_weight
        return matrix

# ---------------------------------------------------------------------------

def QCNNHybridAutoencoderFactory(
    input_dim: int,
    latent_dim: int = 3,
    trash_dim: int = 2,
) -> QCNNHybridAutoencoder:
    """Return a configured hybrid quantum autoencoder."""
    return QCNNHybridAutoencoder(input_dim, latent_dim=latent_dim, trash_dim=trash_dim)

# ---------------------------------------------------------------------------

__all__ = [
    "QCNNHybridAutoencoder",
    "QCNNHybridAutoencoderFactory",
]
