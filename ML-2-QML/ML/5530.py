import torch
import torch.nn as nn

class QuantumKernelMethod(nn.Module):
    """Hybrid classical kernel method combining a transformer encoder and a classical RBF kernel.

    The class can be instantiated with ``kernel_type='classical'`` (default) to use an
    RBF kernel over transformer embeddings, or with ``kernel_type='quantum'`` to
    signal that the quantum implementation should be used.  The quantum
    implementation is provided in the separate ``qml_code`` block.
    """

    def __init__(self,
                 input_dim: int,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 kernel_type: str = "classical",
                 gamma: float = 1.0):
        super().__init__()
        self.kernel_type = kernel_type
        self.gamma = gamma

        # Transformer encoder for feature extraction
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=embed_dim * 4,
                                                   dropout=0.1,
                                                   activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if self.kernel_type!= "classical":
            raise ValueError("Quantum kernel_type is only available in the QML module.")

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project raw input and run through transformer encoder.
        ``x`` has shape ``(batch, seq_len, input_dim)``.
        Returns a batch of embeddings of shape ``(batch, embed_dim)``.
        """
        x = self.input_proj(x)
        # Transformer expects shape (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, embed_dim)
        # Simple pooling: mean over sequence dimension
        return x.mean(dim=1)

    def _rbf_kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise RBF kernel between two batches of embeddings.
        """
        diff = a.unsqueeze(1) - b.unsqueeze(0)  # (len(a), len(b), embed_dim)
        dist_sq = (diff * diff).sum(dim=2)
        return torch.exp(-self.gamma * dist_sq)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix between two batches of inputs.
        ``x`` and ``y`` must have shape ``(batch, seq_len, input_dim)``.
        """
        if self.kernel_type!= "classical":
            raise RuntimeError("Quantum kernel computation is only available in the QML module.")
        a = self._transform(x)
        b = self._transform(y)
        return self._rbf_kernel(a, b)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for ``forward``."""
        return self.forward(a, b)

__all__ = ["QuantumKernelMethod"]
