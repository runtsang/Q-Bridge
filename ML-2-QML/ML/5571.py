"""Hybrid self‑attention module combining classical multi‑head attention
and a lightweight quantum‑inspired attention wrapper.

The class exposes a uniform interface that can be dropped into a
transformer or a regression pipeline.  The classical implementation
uses PyTorch's MultiheadAttention while the quantum implementation
is provided in the separate `qml` module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSelfAttention(nn.Module):
    """Hybrid self‑attention layer supporting classical and quantum modes.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the token embeddings.
    num_heads : int, default 1
        Number of attention heads for the classical mode.
    dropout : float, default 0.1
        Drop‑out probability.
    mode : str, default 'classical'
        ``'classical'`` uses a PyTorch MultiheadAttention.
        ``'quantum'`` is a placeholder for the quantum implementation
        in the `qml` module; attempting to instantiate it here raises
        an error to avoid accidental misuse.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1,
                 dropout: float = 0.1, mode: str = "classical") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mode = mode
        if mode == "classical":
            self.attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
        elif mode == "quantum":
            # The quantum variant is implemented in the qml module.
            raise NotImplementedError(
                "Quantum mode is only available in the qml implementation."
            )
        else:
            raise ValueError(f"Unknown mode {mode!r}")

    def forward(self, inputs: torch.Tensor,
                rotation_params: torch.Tensor | None = None,
                entangle_params: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the self‑attention output.

        For the classical mode the ``rotation_params`` and
        ``entangle_params`` are ignored, mimicking the original
        seed implementation.  They are retained in the signature to
        keep a consistent API with the quantum counterpart.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(batch, seq_len, embed_dim)``.
        rotation_params : torch.Tensor, optional
            Parameters for a quantum rotation layer (ignored here).
        entangle_params : torch.Tensor, optional
            Parameters for a quantum entanglement layer (ignored here).

        Returns
        -------
        torch.Tensor
            The attended representation of shape ``(batch, seq_len, embed_dim)``.
        """
        if self.mode!= "classical":
            raise RuntimeError("Quantum mode not available in the classical module.")
        attn_output, _ = self.attn(inputs, inputs, inputs)
        return attn_output

    @staticmethod
    def generate_superposition_data(num_features: int, samples: int):
        """Utility to generate the synthetic dataset used in the regression
        reference pair.  It is copied verbatim from the ML seed for
        consistency.
        """
        x = torch.rand(samples, num_features, dtype=torch.float32) * 2 - 1
        angles = x.sum(dim=1)
        y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
        return x, y

__all__ = ["HybridSelfAttention"]
