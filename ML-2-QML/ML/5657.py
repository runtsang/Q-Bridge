import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridFunction(torch.autograd.Function):
    """Differentiable interface that forwards a vector of classical values
    through a user supplied quantum circuit and returns the expectation
    value. The circuit must expose a ``run`` method that accepts a NumPy
    array and returns a scalar expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        numpy_inputs = inputs.detach().cpu().numpy()
        exp_vals = ctx.circuit.run(numpy_inputs)
        exp_tensor = torch.tensor(exp_vals, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, exp_tensor)
        return exp_tensor

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.detach().cpu().numpy()) * ctx.shift
        grads = []
        for inp, sh in zip(inputs.detach().cpu().numpy(), shift):
            exp_right = ctx.circuit.run([inp + sh])
            exp_left = ctx.circuit.run([inp - sh])
            grads.append(exp_right - exp_left)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Wrapper that forwards a 1‑D tensor through a quantum circuit."""
    def __init__(self, circuit, shift: float = 0.0):
        super().__init__()
        self.circuit = circuit
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class UnifiedSelfAttentionHybrid(nn.Module):
    """Classical multi‑head self‑attention block that optionally injects a
    quantum circuit into the attention score computation. When ``use_quantum``
    is ``True`` the flattened attention scores are passed through a
    ``Hybrid`` layer, yielding differentiable expectation values that replace
    the raw soft‑max weights."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 quantum_circuit=None,
                 shift: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.use_quantum = use_quantum
        if use_quantum:
            if quantum_circuit is None:
                raise ValueError("quantum_circuit must be provided when use_quantum=True")
            self.hybrid = Hybrid(quantum_circuit, shift)
        else:
            self.hybrid = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, embed_dim)
        Returns:
            Tensor of shape (batch, seq_len, embed_dim)
        """
        batch, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = self.dropout(scores)

        if self.use_quantum:
            scores_flat = scores.reshape(batch * self.num_heads, seq_len * seq_len)
            flat_inputs = scores_flat.flatten()
            quantum_scores = self.hybrid(flat_inputs)
            scores = quantum_scores.reshape(batch, self.num_heads, seq_len, seq_len)

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        return output

__all__ = ["HybridFunction", "Hybrid", "UnifiedSelfAttentionHybrid"]
