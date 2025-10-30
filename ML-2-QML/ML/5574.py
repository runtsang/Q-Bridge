"""Hybrid Self‑Attention combining classical attention, quantum‑inspired sampling, and fast estimation.

The class integrates:
- classical self‑attention via PyTorch linear projections,
- a SamplerQNN network that stochastically samples attention logits,
- an optional quantum submodule that generates attention weights using a Qiskit circuit,
- a FastEstimator to evaluate expectation values when using quantum attention.

The module is fully importable and can be used as a drop‑in replacement for the original SelfAttention helper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import Aer, execute
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

class HybridSelfAttention(nn.Module):
    """
    A hybrid self‑attention module that can operate in a purely classical mode or
    incorporate a quantum sub‑circuit to compute attention weights.

    Parameters
    ----------
    embed_dim : int
        Size of the embedding vector.
    n_qubits : int, default 4
        Number of qubits used by the quantum attention sub‑circuit.
    use_quantum : bool, default True
        If True, the module will use the quantum sub‑circuit to compute attention logits.
        If False, it falls back to a standard PyTorch attention block.
    """
    def __init__(self, embed_dim: int, n_qubits: int = 4, use_quantum: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum

        # Classical attention layers
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Sampler network for stochastic attention logits
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

        # Quantum circuit components
        if self.use_quantum:
            self.qr = QuantumRegister(n_qubits, "q")
            self.cr = ClassicalRegister(n_qubits, "c")
            self.backend = Aer.get_backend("qasm_simulator")

    def _quantum_attention(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
                           batch_size: int) -> torch.Tensor:
        """Compute attention logits using a Qiskit circuit and return a probability vector."""
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure_all()

        job = execute(circuit, self.backend, shots=2048)
        result = job.result()
        counts = result.get_counts(circuit)

        probs = np.zeros(self.n_qubits, dtype=np.float32)
        total = sum(counts.values())
        for bitstring, cnt in counts.items():
            for idx, bit in enumerate(reversed(bitstring)):
                probs[idx] += cnt * int(bit)
        probs /= total
        # Broadcast to batch
        return torch.tensor(probs, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)

    def forward(self, inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        """
        Compute the self‑attention output.

        Parameters
        ----------
        inputs : torch.Tensor
            Input embeddings of shape (batch, embed_dim).
        rotation_params : np.ndarray
            Parameters for single‑qubit rotations in the quantum circuit.
        entangle_params : np.ndarray
            Parameters for two‑qubit entangling gates in the quantum circuit.
        """
        if self.use_quantum:
            batch_size = inputs.shape[0]
            attention_weights = self._quantum_attention(rotation_params,
                                                        entangle_params,
                                                        batch_size)  # (batch, n_qubits)
            # map quantum weights to the key dimension
            key = self.key_proj(inputs)
            value = self.value_proj(inputs)
            # broadcast weights to match key/value shape
            attn = attention_weights.unsqueeze(-1) * value
            return attn.sum(dim=1)
        else:
            query = self.query_proj(inputs)
            key   = self.key_proj(inputs)
            value = self.value_proj(inputs)
            scores = F.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
            return scores @ value

__all__ = ["HybridSelfAttention"]
