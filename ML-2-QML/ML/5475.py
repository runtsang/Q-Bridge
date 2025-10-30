"""Hybrid self‑attention model combining classical and quantum primitives.

This module implements the classical side of the pipeline.  It uses the
SelfAttention class from the base seed, a classical feed‑forward
classifier, a classical RBF kernel and a FastBaseEstimator for
expectation‑like evaluation.  The interface is deliberately
identical to the quantum variant so that the two can be swapped
at runtime.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Iterable, Sequence, Callable, List

from.SelfAttention import SelfAttention as ClassicalSelfAttention
from.QuantumClassifierModel import build_classifier_circuit as build_cls
from.QuantumKernelMethod import kernel_matrix as kernel_matrix_func
from.FastBaseEstimator import FastBaseEstimator


class HybridSelfAttentionModel:
    """Classical hybrid self‑attention pipeline.

    Parameters
    ----------
    embed_dim
        Dimensionality of the embedding used by the attention block.
    num_features
        Number of features expected by the classifier.
    depth
        Depth of the classifier network.
    use_quantum
        When ``True`` the attention block is executed on a Qiskit
        backend; otherwise a NumPy implementation is used.
    backend
        Qiskit backend passed to the quantum attention when
        ``use_quantum=True``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_features: int,
        depth: int,
        use_quantum: bool = False,
        backend=None,
    ) -> None:
        self.embed_dim = embed_dim
        self.num_features = num_features
        self.depth = depth
        self.use_quantum = use_quantum
        self.backend = backend

        # Attention block
        self.attention = ClassicalSelfAttention(embed_dim)

        if use_quantum:
            # Import the quantum variant lazily to avoid unnecessary
            # dependency when the classical path is used.
            from.SelfAttention import SelfAttention as QuantumSelfAttention

            self.quantum_attention = QuantumSelfAttention()
        else:
            self.quantum_attention = None

        # Classifier
        self.classifier, self.cls_encoding, self.cls_weights, self.cls_obs = build_cls(
            num_features, depth
        )

        # Kernel
        self.kernel = kernel_matrix_func

        # Estimator
        self.estimator = FastBaseEstimator(self.classifier)

    def forward(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the full pipeline and return intermediate results."""
        # Attention
        if self.use_quantum:
            counts = self.quantum_attention.run(
                self.backend,
                rotation_params,
                entangle_params,
                shots=shots or 1024,
            )
            probs = np.array(
                [counts.get(bit, 0) for bit in sorted(counts)]
            )
            attention_out = probs / probs.sum() if probs.sum() else probs
        else:
            attention_out = self.attention.run(
                rotation_params, entangle_params, inputs
            )

        # Classifier
        cls_output = self.classifier(
            torch.as_tensor(inputs, dtype=torch.float32)
        ).numpy()

        # Kernel matrix between attention and classifier outputs
        kernel_mat = self.kernel(attention_out, cls_output)

        return {
            "attention": attention_out,
            "classifier": cls_output,
            "kernel": kernel_mat,
        }

    def estimate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]] | None = None,
    ) -> List[List[float]]:
        """Convenience wrapper around the FastBaseEstimator."""
        return self.estimator.evaluate(observables or [], parameter_sets)
