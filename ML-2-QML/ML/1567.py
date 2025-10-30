"""Extended multi‑head self‑attention with residual connections and trainable projection matrices.

The public API mirrors the reference but augments the core logic so that
* the input is assumed to be a batch of sequences (shape `(batch, seq_len, embed_dim)`).
* a small trainable weight matrix is added to the scaled dot‑product
  (the `query`, `key`, **and** value).  The weight matrix is initialized
  with a normal distribution and is learnable via PyTorch.
* the attention is computed per head and the heads are concatenated.
* an optional residual connection adds the original input back to the
  output.
"""

import numpy as np
import torch

def SelfAttention():
    class ClassicalSelfAttention:
        def __init__(self, embed_dim: int, num_heads: int = 1):
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            if embed_dim % num_heads!= 0:
                raise ValueError("embed_dim must be divisible by num_heads")
            self.head_dim = embed_dim // num_heads
            # Trainable projection used by all projections
            self.W_proj = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))
            # Projection for query, key, value
            self.W_q = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))
            self.W_k = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))
            self.W_v = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            residual: bool = True,
        ) -> np.ndarray:
            """
            Parameters
            ----------
            rotation_params : np.ndarray
                Weight matrix (embed_dim, embed_dim) used to compute queries.
            entangle_params : np.ndarray
                Weight matrix (embed_dim, embed_dim) used to compute keys.
            inputs : np.ndarray
                Input tensor of shape (batch, seq_len, embed_dim).
            residual : bool, optional
                If True, add the original input to the output.

            Returns
            -------
            np.ndarray
                The self‑attention output of shape (batch, seq_len, embed_dim).
            """
            # Convert inputs and parameters to torch tensors
            inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
            rot_t = torch.as_tensor(rotation_params, dtype=torch.float32)
            ent_t = torch.as_tensor(entangle_params, dtype=torch.float32)

            # Project inputs
            proj = torch.matmul(inputs_t, self.W_proj)
            # Compute Q, K, V
            q = torch.matmul(proj, rot_t)
            k = torch.matmul(proj, ent_t)
            v = torch.matmul(proj, self.W_v)

            # Reshape for multi‑head
            batch, seq_len, _ = q.shape
            q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Scaled dot‑product attention
            scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim), dim=-1)
            context = torch.matmul(scores, v)  # (batch, heads, seq_len, head_dim)

            # Concatenate heads
            context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

            if residual:
                context = context + inputs_t

            return context.detach().numpy()

    return ClassicalSelfAttention(embed_dim=4, num_heads=1)

__all__ = ["SelfAttention"]
