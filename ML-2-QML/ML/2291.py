"""Hybrid self‑attention module that fuses a classical transformer head with a quantum sampler.

The class exposes a PyTorch ``nn.Module`` that:
* builds query/key/value projections,
* delegates the attention weight computation to a parameterised quantum sampler,
* multiplies the classical attention scores with the quantum‑derived distribution,
* produces a weighted sum of the value vectors.

This structure preserves the scalability of a transformer while injecting a
non‑classical source of randomness that can be tuned via the quantum circuit
parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler

class SelfAttentionGen093(nn.Module):
    """Hybrid self‑attention head combining classical linear layers with a quantum sampler."""

    def __init__(self, embed_dim: int, n_qubits: int = 4, shots: int = 1024):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings. Must equal ``n_qubits`` to keep
            the attention score matrix square.
        n_qubits : int, optional
            Number of qubits used in the quantum sampler, by default 4.
        shots : int, optional
            Number of shots for the quantum sampler, by default 1024.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.shots = shots

        # Classical linear projections
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        # Quantum sampler that produces a probability distribution over ``n_qubits`` outcomes
        self.sampler_qnn = self._build_sampler_qnn()

    def _build_sampler_qnn(self) -> SamplerQNN:
        """Build a parameterised quantum circuit that outputs a qubit‑wise probability distribution."""
        # Parameter vectors for inputs and rotations
        input_params = ParameterVector("input", self.n_qubits)
        weight_params = ParameterVector("weight", self.n_qubits * 3)

        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(input_params[i], i)
            qc.rx(weight_params[3 * i], i)
            qc.ry(weight_params[3 * i + 1], i)
            qc.rz(weight_params[3 * i + 2], i)

        # Entangle neighbouring qubits to introduce non‑local correlations
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)

        sampler = StatevectorSampler()
        return SamplerQNN(circuit=qc,
                          input_params=input_params,
                          weight_params=weight_params,
                          sampler=sampler)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid self‑attention output.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(batch, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, embed_dim)``.
        """
        batch_size = inputs.size(0)

        # Classical projections
        q = self.query(inputs)          # (B, D)
        k = self.key(inputs)            # (B, D)
        v = self.value(inputs)          # (B, D)

        # Classical attention scores (B, D)
        scores = torch.softmax((q @ k.t()) / np.sqrt(self.embed_dim), dim=-1)

        # Quantum‑derived probability distribution per sample
        quantum_weights = []
        for b in range(batch_size):
            # Random rotation parameters for the sampler
            weight_params = np.random.randn(self.n_qubits * 3)
            # Map the input embedding to rotation angles for the input parameters
            input_angles = inputs[b].detach().cpu().numpy()
            if input_angles.shape[0]!= self.n_qubits:
                raise ValueError("Input embedding size must match the number of qubits.")
            counts = self.sampler_qnn.sample(input_angles,
                                            weight_params,
                                            shots=self.shots)
            probs = np.array([counts.get(bin(i), 0) / self.shots for i in range(self.n_qubits)])
            quantum_weights.append(torch.tensor(probs, dtype=torch.float32, device=inputs.device))

        quantum_weights = torch.stack(quantum_weights)      # (B, D)

        # Combine classical scores with quantum weights
        combined_scores = scores * quantum_weights

        # Weighted sum of value vectors
        output = torch.matmul(combined_scores, v)

        return output

__all__ = ["SelfAttentionGen093"]
