import torch
import torch.nn as nn
import numpy as np

class HybridAttentionModel(nn.Module):
    """
    Classical self‑attention module that mirrors a quantum‑style parameter
    layout.  The rotation_params and entangle_params are supplied as 2‑D
    numpy arrays with shapes (embed_dim, embed_dim).  Parameters can be
    clipped to a bounded range to keep the optimisation stable – a
    technique borrowed from the fraud‑detection seed.

    The forward pass performs a standard scaled dot‑product attention
    but uses the supplied rotation_params for the query weight matrix
    and entangle_params for the key weight matrix.  The value matrix
    is learned independently.  Dropout can be applied to the attention
    weights if desired.
    """

    def __init__(self, embed_dim: int, clip: bool = True, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.clip = clip
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialise linear layers with identity weights; the real
        # parameters will be injected in forward().
        self.query_layer = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_layer   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_layer = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute self‑attention.

        Args:
            inputs:  (batch, embed_dim) tensor of feature vectors.
            rotation_params: (embed_dim, embed_dim) rotation matrix.
            entangle_params: (embed_dim, embed_dim) key matrix.

        Returns:
            Tensor of shape (batch, embed_dim) – the attended representation.
        """
        if self.clip:
            rotation_params = np.clip(rotation_params, -5.0, 5.0)
            entangle_params = np.clip(entangle_params, -5.0, 5.0)

        # Inject the supplied parameters into the linear layers.
        with torch.no_grad():
            self.query_layer.weight.copy_(torch.from_numpy(rotation_params).float())
            self.key_layer.weight.copy_(torch.from_numpy(entangle_params).float())

        queries = self.query_layer(inputs)
        keys    = self.key_layer(inputs)
        values  = self.value_layer(inputs)

        scores = torch.softmax(
            queries @ keys.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1
        )
        scores = self.dropout(scores)

        return scores @ values

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Convenience wrapper that accepts plain numpy arrays and returns
        a numpy array.  Useful for quick prototyping or for the hybrid
        interface that expects a classical run method.
        """
        inputs_t = torch.tensor(inputs, dtype=torch.float32)
        out_t = self.forward(inputs_t, rotation_params, entangle_params)
        return out_t.detach().cpu().numpy()
