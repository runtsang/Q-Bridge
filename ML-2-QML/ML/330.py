import torch
import numpy as np

class SelfAttention:
    """Classical self‑attention block with learnable projections."""
    def __init__(self, embed_dim: int, bias: bool = True):
        self.embed_dim = embed_dim
        self.query_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_proj   = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
    
    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Compute self‑attention over the input sequence."""
        x = torch.as_tensor(inputs, dtype=torch.float32)
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out.detach().numpy()
