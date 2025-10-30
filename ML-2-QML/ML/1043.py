import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

class SelfAttentionModule(nn.Module):
    """
    Hybrid classical‑quantum self‑attention block.

    The block contains two parallel branches:

    * Classical branch – linear projections for query, key and value
      followed by a soft‑max attention map.
    * Quantum branch – a variational circuit that outputs a probability
      distribution over the sequence positions.  The circuit is
      parameterised by rotation angles and entangling gates and
      executed on a state‑vector simulator.  The resulting distribution
      is used to weight the value vectors.

    The two attention maps are fused via a learnable scalar ``alpha``.
    """

    def __init__(self, embed_dim: int, n_qbits: int = 4, device: str = "cpu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qbits = n_qbits
        self.device = device

        # Classical branch
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Fusion coefficient
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Quantum backend
        self.backend = Aer.get_backend("statevector_simulator")

    def _build_quantum_circuit(self, seq_len: int, rotation_params: np.ndarray,
                               entangle_params: np.ndarray):
        """
        Construct a variational circuit that produces a probability
        distribution over ``seq_len`` positions.
        """
        qc = QuantumCircuit(seq_len)
        # Rotation gates
        for i in range(seq_len):
            params = rotation_params[3 * i: 3 * i + 3]
            qc.rx(params[0], i)
            qc.ry(params[1], i)
            qc.rz(params[2], i)
        # Entangling gates
        for i in range(seq_len - 1):
            qc.cx(i, i + 1)
            qc.rz(entangle_params[i], i)
        return qc

    def _quantum_attention(self, seq_len: int, rotation_params: np.ndarray,
                           entangle_params: np.ndarray):
        """
        Execute the circuit and return a soft‑max distribution over
        sequence positions.
        """
        qc = self._build_quantum_circuit(seq_len, rotation_params, entangle_params)
        job = execute(qc, self.backend, shots=1024)
        result = job.result()
        probs = result.get_counts(qc)
        # Convert counts to probabilities
        prob_vec = np.zeros(seq_len)
        for bitstring, count in probs.items():
            idx = int(bitstring[::-1], 2) % seq_len
            prob_vec[idx] += count
        prob_vec = prob_vec / prob_vec.sum()
        return torch.tensor(prob_vec, dtype=torch.float32, device=self.device)

    def run(self, inputs: torch.Tensor,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray):
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, seq_len, embed_dim)
        rotation_params : np.ndarray
            Shape (seq_len, 3)
        entangle_params : np.ndarray
            Shape (seq_len-1,)
        Returns
        -------
        output : torch.Tensor
            Shape (batch, seq_len, embed_dim)
        """
        batch, seq_len, _ = inputs.shape

        # Classical attention
        q = self.query_proj(inputs)          # (B, S, D)
        k = self.key_proj(inputs)            # (B, S, D)
        v = self.value_proj(inputs)          # (B, S, D)
        scores = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.embed_dim), dim=-1)
        classical_out = torch.matmul(scores, v)   # (B, S, D)

        # Quantum attention (same distribution for all batch elements)
        quantum_dist = self._quantum_attention(seq_len, rotation_params, entangle_params)
        quantum_dist = quantum_dist.unsqueeze(0).unsqueeze(-1)  # (1, S, 1)
        quantum_out = v * quantum_dist  # (B, S, D)

        # Fusion
        output = self.alpha * classical_out + (1 - self.alpha) * quantum_out
        return output

__all__ = ["SelfAttentionModule"]
