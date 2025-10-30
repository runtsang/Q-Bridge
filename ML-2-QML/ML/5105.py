import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Classical self‑attention block inspired by the original SelfAttention helper
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, inputs: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        # Linear projections for query, key, and value
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key   = inputs @ entangle_params.reshape(self.embed_dim, -1)
        value = inputs
        # Scaled dot‑product attention
        scores = F.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

# Hybrid sampler that uses the attention block and a small feed‑forward network
class SamplerQNN(nn.Module):
    """
    Classical sampler network with optional quantum sampling support.
    The network first applies a self‑attention block, then a lightweight
    feed‑forward head that outputs a soft‑max distribution.  A quantum
    sampler can be called via the `sample_quantum` method.
    """
    def __init__(self, embed_dim: int = 4, hidden_dim: int = 8):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        # Parameters that will be fed to the quantum sampler
        self.quantum_params = nn.Parameter(torch.randn(embed_dim, 3, dtype=torch.float64))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim - 1, dtype=torch.float64))
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(inputs, self.quantum_params, self.entangle_params)
        logits = self.fc(attn_out)
        return F.softmax(logits, dim=-1)

    def sample_quantum(self, angles: np.ndarray, shots: int = 1024) -> float:
        """
        Execute a simple two‑qubit parameterised circuit and return the
        probability of measuring |11⟩.  The angles are interpreted as
        rotation parameters for the two qubits.
        """
        from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
        qc = QuantumCircuit(2)
        qc.ry(angles[0], 0)
        qc.ry(angles[1], 1)
        qc.cx(0, 1)
        qc.measure_all()
        backend = Aer.get_backend("qasm_simulator")
        transpiled = transpile(qc, backend)
        qobj = assemble(transpiled, shots=shots)
        result = execute(qc, backend, shots=shots).result()
        counts = result.get_counts()
        prob_11 = counts.get('11', 0) / shots
        return prob_11

__all__ = ["SamplerQNN"]
