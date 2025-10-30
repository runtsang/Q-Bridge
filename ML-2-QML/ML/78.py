import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalSelfAttention(nn.Module):
    """
    Hybrid classical self‑attention block that mirrors the original interface.
    The rotation and entangle parameters are interpreted as the weight
    matrices for the query and key projections.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q = self.to_q(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        return self.out_proj(attn)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Exposes the original ``run`` signature.  ``rotation_params`` and
        ``entangle_params`` are reshaped and written into the weight matrices
        of the Q and K linear layers respectively.  The method then forwards
        the input through the attention block and returns a NumPy array.
        """
        # Load parameters into the linear layers
        q_weight = torch.from_numpy(rotation_params.reshape(self.embed_dim, self.embed_dim)).float()
        k_weight = torch.from_numpy(entangle_params.reshape(self.embed_dim, self.embed_dim)).float()
        with torch.no_grad():
            self.to_q.weight.copy_(q_weight)
            self.to_k.weight.copy_(k_weight)

        x = torch.from_numpy(inputs).float()
        out = self.forward(x)
        return out.detach().cpu().numpy()

def SelfAttention() -> ClassicalSelfAttention:
    """
    Factory that returns an instance compatible with the original API.
    """
    return ClassicalSelfAttention(embed_dim=4)

__all__ = ["SelfAttention"]
