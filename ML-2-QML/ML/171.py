import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHybrid(nn.Module):
    """
    Multi‑head self‑attention with optional quantum embedding.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads.
    use_quantum : bool
        If True, the key projection is computed by a quantum circuit
        (see the qml module).  The quantum parameters are treated as
        learnable torch tensors and passed to the circuit during the
        forward pass.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, use_quantum: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_quantum = use_quantum

        # Classical linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # If quantum is enabled, initialise quantum parameters
        if self.use_quantum:
            # We expose two sets of parameters: rotation and entangle,
            # each of shape (num_qubits, 3) and (num_qubits-1,)
            # For simplicity we use 4 qubits per head.
            self.num_qubits = 4
            self.rotation_params = nn.Parameter(
                torch.rand(self.num_qubits, 3, dtype=torch.float32)
            )
            self.entangle_params = nn.Parameter(
                torch.rand(self.num_qubits - 1, dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = x.size()

        # Linear projections
        Q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Optionally replace K with a quantum‑generated matrix
        if self.use_quantum:
            from.qml import QuantumSelfAttention  # local import to avoid circular
            quantum_attn = QuantumSelfAttention(self.num_qubits)
            k_list = []
            for _ in range(batch):
                counts = quantum_attn.run(
                    rotation_params=self.rotation_params.detach().cpu().numpy(),
                    entangle_params=self.entangle_params.detach().cpu().numpy(),
                    shots=512,
                )
                probs = np.zeros(self.num_qubits)
                for state, cnt in counts.items():
                    idx = int(state, 2)
                    probs[idx] = cnt
                probs /= probs.sum() + 1e-8
                k_vec = torch.from_numpy(probs).float().to(x.device)
                k_list.append(k_vec)
            K = torch.stack(k_list, dim=0).unsqueeze(1).repeat(1, seq_len, 1, 1)
            K = K.view(batch, seq_len, self.embed_dim)

        # Reshape for multi‑head
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.out_proj(attn_output)
        return out

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Legacy interface retained for backward compatibility.
        It simply forwards to the modern `forward` method after
        wrapping inputs in a torch tensor.
        """
        x = torch.from_numpy(inputs).float()
        return self.forward(x).detach().cpu().numpy()

__all__ = ["SelfAttentionHybrid"]
