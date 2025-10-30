import torch
import numpy as np
from torch import nn

class HybridQuantumAttentionLayer(nn.Module):
    """
    Hybrid classical layer combining a fully connected transformation with a self‑attention
    mechanism.  The fully connected part uses a single linear mapping whose weights can
    be supplied explicitly (mimicking the quantum parameter vector).  The attention part
    uses learnable query/key/value matrices that are re‑parameterized by the user.
    """

    def __init__(self, embed_dim: int = 4, fc_input_dim: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Linear(fc_input_dim, embed_dim, bias=False)

        # Attention parameters (treated as rotation/entangle parameters)
        self.query_w = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.key_w   = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.value_w = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def run(
        self,
        fc_thetas: np.ndarray | torch.Tensor,
        rotation_params: np.ndarray | torch.Tensor,
        entangle_params: np.ndarray | torch.Tensor,
        inputs: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """
        Execute the hybrid layer.

        Parameters
        ----------
        fc_thetas : array-like
            Flat array of length `embed_dim` used to overwrite the linear layer weights.
        rotation_params : array-like
            Flattened matrix of shape (embed_dim, embed_dim) used as query weights.
        entangle_params : array-like
            Flattened matrix of shape (embed_dim, embed_dim) used as key weights.
        inputs : array-like
            Input matrix of shape (batch, fc_input_dim).

        Returns
        -------
        numpy.ndarray
            Output of the attention mechanism.
        """
        # Update FC weights
        fc_w = torch.as_tensor(fc_thetas, dtype=torch.float32).reshape(self.fc.weight.shape)
        self.fc.weight.data = fc_w

        # Update attention weights
        self.query_w.data = torch.as_tensor(rotation_params, dtype=torch.float32).reshape(self.query_w.shape)
        self.key_w.data   = torch.as_tensor(entangle_params, dtype=torch.float32).reshape(self.key_w.shape)

        # Forward pass
        x = torch.as_tensor(inputs, dtype=torch.float32)
        fc_out = torch.tanh(self.fc(x))

        queries = fc_out @ self.query_w
        keys    = fc_out @ self.key_w
        values  = fc_out @ self.value_w

        scores = torch.softmax(queries @ keys.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        output = scores @ values
        return output.detach().numpy()
