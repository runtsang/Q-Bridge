"""Hybrid quanvolution implemented with Qiskit, combining a quantum filter with a classical classifier.  Includes FastBaseEstimator‑style evaluation and shot noise simulation."""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Pauli
from qiskit.circuit.library import RandomEntangler
from collections.abc import Iterable, Sequence
from typing import List, Tuple
from qiskit.quantum_info.operators import PauliSumOp

# Utility type for observables
BaseOperator = PauliSumOp

def _ensure_batch(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

class QuanvolutionHybrid:
    """Quantum quanvolution filter with classical linear classifier.

    The filter processes 2×2 image patches on 4 qubits using a random entangling layer.
    Results are expectation values of PauliZ, aggregated into a feature vector.
    A classical linear layer produces logits.  The class exposes evaluate and
    evaluate_with_noise methods following the FastBaseEstimator API.
    """

    def __init__(self, n_classes: int = 10, patch_dim: int = 2, n_q_features: int = 4) -> None:
        self.patch_dim = patch_dim
        self.n_q_features = n_q_features
        self.n_patches = (28 // patch_dim) ** 2
        self.n_features = n_q_features * self.n_patches
        # Classical linear classifier weights
        self.weights = np.random.randn(self.n_features, n_classes).astype(np.float64)
        self.bias = np.random.randn(n_classes).astype(np.float64)

        # Quantum subcircuit template
        self._circuit_template = self._build_patch_circuit()

    def _build_patch_circuit(self) -> QuantumCircuit:
        """Build a template circuit for a 2×2 patch."""
        qr = QuantumRegister(self.n_q_features)
        qc = QuantumCircuit(qr)
        # Parameterized Ry for each pixel
        for i in range(self.n_q_features):
            qc.ry(Parameter(f"ry_{i}"), qr[i])
        # Random entangler
        entangler = RandomEntangler(
            n_qubits=self.n_q_features,
            entanglement_type="full",
            repetitions=2,
        )
        qc.append(entangler, qr)
        return qc

    def _apply_patch(self, patch: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Apply the patch circuit to a single patch and return expectation values."""
        qc = self._circuit_template.copy()
        # Bind Ry parameters to pixel values
        for i, val in enumerate(params):
            qc.assign_parameters({qc.parameters[i]: val}, inplace=True)
        sv = Statevector.from_instruction(qc)
        exp_vals = np.array([sv.expectation_value(Pauli("Z", [i])) for i in range(self.n_q_features)])
        return exp_vals

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Compute logits for a single 28×28 grayscale image."""
        # Extract patches
        patches = []
        for r in range(0, 28, self.patch_dim):
            for c in range(0, 28, self.patch_dim):
                patch = image[r : r + self.patch_dim, c : c + self.patch_dim].flatten()
                patches.append(patch)
        # Compute quantum features
        features = []
        for patch in patches:
            exp = self._apply_patch(patch, patch)  # use pixel values as parameters
            features.extend(exp)
        features = np.array(features, dtype=np.float64).reshape(1, -1)
        logits = features @ self.weights + self.bias
        # Log‑softmax
        log_softmax = -np.log(np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))
        return log_softmax[0]

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each observable over batches of images.

        Parameters
        ----------
        observables : iterable of BaseOperator
            PauliSumOp objects to evaluate.
        parameter_sets : sequence of image arrays
            Each inner sequence is a flattened 784‑size image.

        Returns
        -------
        List of lists of complex numbers
            One row per image, one column per observable.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            image = np.asarray(params, dtype=np.float32).reshape(28, 28)
            obs_accum = np.zeros(len(observables), dtype=np.complex128)
            for r in range(0, 28, self.patch_dim):
                for c in range(0, 28, self.patch_dim):
                    patch = image[r : r + self.patch_dim, c : c + self.patch_dim].flatten()
                    qc = self._circuit_template.copy()
                    for i, val in enumerate(patch):
                        qc.assign_parameters({qc.parameters[i]: val}, inplace=True)
                    sv = Statevector.from_instruction(qc)
                    for idx, obs in enumerate(observables):
                        obs_accum[idx] += sv.expectation_value(obs)
            obs_avg = obs_accum / self.n_patches
            results.append(obs_avg.tolist())
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Add shot noise to expectation value evaluation."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                rng.normal(np.real(val), max(1e-6, 1 / shots)) + 1j * rng.normal(np.imag(val), max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

__all__ = ["QuanvolutionHybrid"]
