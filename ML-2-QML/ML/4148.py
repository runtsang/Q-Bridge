import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridLayer(nn.Module):
    """
    Classical hybrid fully‑connected layer that incorporates a self‑attention mechanism
    and a dense output.  The attention weights are learned via a trainable
    rotation matrix; the entangle parameters are present in the signature for
    API compatibility but are not used in the classical version.
    """

    def __init__(self, input_dim: int, output_dim: int, n_qubits: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits

        # Linear transformation after attention
        self.linear = nn.Linear(input_dim, output_dim)

        # Self‑attention parameters (learnable)
        self.rotation_params = nn.Parameter(torch.randn(input_dim, input_dim))
        self.entangle_params = nn.Parameter(torch.randn(n_qubits - 1))

    def run(
        self,
        rotation_params: torch.Tensor | None,
        entangle_params: torch.Tensor | None,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the output of the hybrid layer.

        Args:
            rotation_params: Optional matrix of shape (input_dim, input_dim).
                             If None, the internally stored parameters are used.
            entangle_params: Optional vector of length n_qubits-1.
                             Ignored in the classical implementation.
            inputs: Tensor of shape (batch, input_dim).

        Returns:
            Tensor of shape (batch, output_dim).
        """
        if rotation_params is None:
            rotation_params = self.rotation_params
        # Self‑attention
        query = torch.matmul(inputs, rotation_params)
        key = torch.matmul(inputs, rotation_params.t())
        scores = F.softmax(
            torch.matmul(query, key.t()) / np.sqrt(self.input_dim), dim=-1
        )
        weighted = torch.matmul(scores, inputs)
        return self.linear(weighted)

__all__ = ["HybridLayer"]
