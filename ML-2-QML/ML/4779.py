"""Unified self‑attention model combining classical attention, QCNN feature extraction, and fast evaluation.

This module defines :class:`UnifiedSelfAttentionModel`, a PyTorch module that
merges the classical self‑attention block from the original SelfAttention
seed with a QCNN‑style feature extractor.  The class exposes a single
``forward`` method that accepts a batch of embeddings, optional rotation
and entanglement parameters, and a flag to switch between the classical
and quantum branches (the quantum branch is handled by the QML module).
A lightweight fast estimator is wrapped around the classifier head to
enable quick evaluation of batches or parameter sweeps.
"""

import numpy as np
import torch
from torch import nn

# Import the classical helpers from the seed files
# (Assumes the seed modules are in the same package)
from.SelfAttention import ClassicalSelfAttention
from.QCNN import QCNNModel
from.FastBaseEstimator import FastBaseEstimator

__all__ = ["UnifiedSelfAttentionModel"]

class UnifiedSelfAttentionModel(nn.Module):
    """
    Hybrid classical self‑attention model with QCNN feature extraction.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    qcnn_output_dim : int
        Output dimensionality of the QCNN feature extractor.
    """
    def __init__(self, embed_dim: int = 4, qcnn_output_dim: int = 8) -> None:
        super().__init__()
        # Classical self‑attention block
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)

        # QCNN feature extractor (mirrors the QCNNModel seed)
        self.qcnn = QCNNModel()

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(qcnn_output_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Fast estimator that can evaluate the head on a batch of inputs
        self.estimator = FastBaseEstimator(self.head)

    def forward(self,
                inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                use_quantum: bool = False) -> torch.Tensor:
        """
        Forward pass for the hybrid model.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        rotation_params : np.ndarray
            Rotation parameters for the classical attention block.
        entangle_params : np.ndarray
            Entanglement parameters for the classical attention block.
        use_quantum : bool, optional
            If ``True`` the method delegates to the quantum branch
            (implemented in the QML module).  The quantum branch
            simply returns the raw QCNN logits for the given
            parameters.  The flag is provided for API symmetry
            but the actual quantum execution resides in the
            :mod:`qml_code` module.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, 1).
        """
        # Flatten batch and sequence for the classical attention block
        batch, seq_len, _ = inputs.shape
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])  # (batch*seq_len, embed_dim)

        # Classical attention
        attn_output = self.attention.run(
            rotation_params=rotation_params,
            entangle_params=entangle_params,
            inputs=flat_inputs.numpy()
        )
        attn_tensor = torch.as_tensor(attn_output, dtype=torch.float32)
        attn_tensor = attn_tensor.reshape(batch, seq_len, -1)

        # Prepare QCNN input: flatten the attention output across the sequence dimension
        qcnn_input = attn_tensor.reshape(batch, -1)  # (batch, seq_len*embed_dim)

        # QCNN feature extraction
        qcnn_features = self.qcnn(qcnn_input)

        # Classification head
        logits = self.head(qcnn_features)
        return logits

    def evaluate(self,
                 observables,
                 parameter_sets,
                 shots: int | None = None) -> list[list[float]]:
        """
        Evaluate a collection of parameters against a set of observables
        using the fast estimator backend.

        Parameters
        ----------
        observables : Iterable[Callable[[torch.Tensor], torch.Tensor | float]]
            Scalar observables that operate on the model output.
        parameter_sets : Sequence[Sequence[float]]
            Sequence of parameter vectors that will be passed to the head.
        shots : int | None, optional
            Number of shots for the estimator.  If ``None`` the estimator
            performs a deterministic evaluation.

        Returns
        -------
        list[list[float]]
            Nested list of scalar results for each parameter set.
        """
        return self.estimator.evaluate(observables, parameter_sets, shots=shots)
