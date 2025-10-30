from __future__ import annotations

import torch
from torch import nn
import numpy as np
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN

class EstimatorQNN(nn.Module):
    """
    Hybrid feed‑forward + self‑attention + latent auto‑encoder network
    with a quantum expectation head.

    The architecture combines:
      - Autoencoder encoder to produce latent features.
      - Classical self‑attention to capture interactions.
      - A variational quantum circuit (RealAmplitudes) that takes the latent
        representation as input rotations and outputs a single expectation
        value.
    """
    def __init__(
        self,
        input_dim: int = 2,
        latent_dim: int = 3,
        attention_dim: int = 4,
        hidden_layers: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()

        # Autoencoder encoder
        if hidden_layers is None:
            hidden_layers = (64, 32)
        encoder_layers = []
        in_dim = input_dim
        for h in hidden_layers:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.1))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Self‑attention
        self.attn = nn.MultiheadAttention(
            embed_dim=attention_dim, num_heads=2, batch_first=True
        )

        # Projection to match qubit count
        self.projection = nn.Linear(attention_dim, latent_dim)

        # Quantum head
        self.quantum_head = self._build_quantum_head(num_qubits=latent_dim)

    def _build_quantum_head(self, num_qubits: int) -> QiskitEstimatorQNN:
        """Build a Qiskit EstimatorQNN instance."""
        # Input parameters for each qubit
        input_params = [Parameter(f"input_{i}") for i in range(num_qubits)]
        # Variational ansatz
        ansatz = RealAmplitudes(num_qubits, reps=2)
        # Compose circuit
        qc = QuantumCircuit(num_qubits)
        for i, p in enumerate(input_params):
            qc.ry(p, i)
        qc.compose(ansatz, range(num_qubits), inplace=True)
        # Observable: sum of PauliZ on all qubits
        observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])
        estimator = StatevectorEstimator()
        return QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=input_params,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x: Tensor of shape (batch, input_dim)
        """
        # Encode input
        z = self.encoder(x)  # (batch, latent_dim)

        # Self‑attention: treat each latent dimension as a token
        seq = z.unsqueeze(1)  # (batch, 1, latent_dim)
        attn_out, _ = self.attn(seq, seq, seq)  # (batch, 1, attention_dim)
        attn_proj = self.projection(attn_out.squeeze(1))  # (batch, latent_dim)

        # Combine latent and attention
        combined = z + attn_proj  # (batch, latent_dim)

        # Quantum expectation: run for each sample in batch
        batch_np = combined.detach().cpu().numpy()
        q_output = []
        for sample in batch_np:
            # EstimatorQNN expects 2D array of shape (n_samples, n_inputs)
            out = self.quantum_head.predict(sample.reshape(1, -1))
            q_output.append(out[0])
        q_output = torch.tensor(q_output, dtype=torch.float32, device=x.device)

        return q_output
