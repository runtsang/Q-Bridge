"""Hybrid quantum estimator with a quanvolutional circuit and linear post‑processing."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.quantum_info import Statevector, Pauli
from qiskit.opflow import PauliSumOp, StateFn, ExpectationExpectation

class QuantumQuanvolutionFilter:
    """Quantum kernel applied to 2×2 patches of a grayscale image."""
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_qubits = 4
        self._random_layer = self._build_random_layer()

    def _build_random_layer(self) -> List[tuple[int, int]]:
        """Return a list of two‑qubit gate pairs for a random layer."""
        pairs = []
        for _ in range(8):
            a, b = self.rng.choice(self.n_qubits, size=2, replace=False)
            pairs.append((a, b))
        return pairs

    def _patch_circuit(self, patch: np.ndarray) -> QuantumCircuit:
        """Return a circuit that encodes a 2×2 patch and applies the random layer."""
        qr = QuantumRegister(self.n_qubits)
        qc = QuantumCircuit(qr)
        # encode pixel values as rotations
        for i, val in enumerate(patch.flatten()):
            qc.ry(val, qr[i])
        # random two‑qubit entangling layer
        for a, b in self._random_layer:
            qc.cx(qr[a], qr[b])
        return qc

    def forward(self, images: np.ndarray) -> np.ndarray:
        """Apply the filter to a batch of images."""
        batch_features: List[np.ndarray] = []
        for img in images:
            # img shape (28,28)
            patches: List[np.ndarray] = []
            for r in range(0, 28, 2):
                for c in range(0, 28, 2):
                    patch = img[r:r+2, c:c+2]
                    qc = self._patch_circuit(patch)
                    # simulate statevector
                    sv = Statevector.from_instruction(qc)
                    # measure all qubits in Z basis
                    meas = sv.expectation_value(Pauli('Z' * self.n_qubits))
                    patches.append(meas)
            batch_features.append(np.concatenate(patches))
        return np.stack(batch_features)


class HybridFastEstimator:
    """Evaluate a quantum‑classical hybrid model for batches of images and observables."""
    def __init__(self, linear_weights: np.ndarray, linear_bias: np.ndarray | None = None) -> None:
        """
        linear_weights shape (num_classes, feature_dim)
        linear_bias shape (num_classes,)
        """
        self.filter = QuantumQuanvolutionFilter()
        self.W = linear_weights
        self.b = linear_bias if linear_bias is not None else np.zeros(self.W.shape[0], dtype=float)

    def evaluate(
        self,
        observables: Iterable[PauliSumOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables) or [PauliSumOp.from_list([("I", 1)])]
        results: List[List[complex]] = []
        for params in parameter_sets:
            # generate features
            features = self.filter.forward(np.array(params))
            # linear post‑processing
            logits = features @ self.W.T + self.b
            row: List[complex] = []
            for obs in observables:
                # treat obs as a dummy scalar operator over the logits
                weight = np.array([float(c) for _, c in obs.primitive.to_list()])
                row.append(complex(logits @ weight))
            results.append(row)
        return results


__all__ = ["HybridFastEstimator"]
