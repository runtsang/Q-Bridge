"""Quantum autoencoder and graph neural network utilities.

This module implements:
* VariationalQuantumAutoEncoder: encodes classical feature vectors into a quantum
  latent statevector of size 2**latent_dim using a RealAmplitudes ansatz and a
  swap‑test for decoding.
* QuantumGraphNeuralNetwork: builds a weighted graph from latent statevectors
  based on fidelity, and provides a method to retrieve adjacency matrices.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import torch
import torch.nn as nn

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector

class VariationalQuantumAutoEncoder(nn.Module):
    """Quantum auto‑encoder based on a RealAmplitudes ansatz and a swap test.

    The encoder maps classical features into a latent statevector of size
    2**latent_dim.  The decoder is a classical linear layer that maps the
    probability distribution of the latent statevector back to the original
    feature dimension.
    """

    def __init__(self,
                 latent_dim: int = 3,
                 num_trash: int = 2,
                 device: torch.device | None = None) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.num_qubits = latent_dim + num_trash + 1  # +1 ancilla for swap test
        self.device = device or torch.device("cpu")

        # Trainable parameters for the ansatz
        self.params = nn.Parameter(torch.randn(self.num_qubits * 3, dtype=torch.float32))

        # Classical decoder to reconstruct the original feature vector
        self.decoder = nn.Linear(2 ** self.latent_dim, 784)

        # Aer simulator
        self.simulator = AerSimulator(method="statevector")

    def _build_circuit(self, features: np.ndarray) -> tuple[QuantumCircuit, RealAmplitudes]:
        """Build a parametric circuit for a single data sample."""
        qr = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(qr)

        # Encode classical data into the first `latent_dim` qubits using a Ry gate
        for i in range(self.latent_dim):
            angle = np.pi * features[i] / 255.0  # normalize pixel values
            qc.ry(angle, qr[i])

        # Append RealAmplitudes ansatz
        ansatz = RealAmplitudes(self.num_qubits, reps=2, insert_barriers=False)
        qc.append(ansatz, qr)

        # Swap test with ancilla
        ancilla = self.num_qubits - 1
        qc.h(ancilla)
        for i in range(self.num_trash):
            qc.cswap(ancilla, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(ancilla)

        return qc, ansatz

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode a batch of features into latent statevectors.

        Parameters
        ----------
        features : torch.Tensor
            Shape (B, 784) flattened image patches.

        Returns
        -------
        latent : torch.Tensor
            Shape (B, 2**latent_dim, 2) complex amplitudes of the latent
            quantum state.  The amplitudes are returned as a real tensor
            with shape (B, 2**latent_dim, 2) where the last dimension
            contains real and imaginary parts.
        """
        batch_size = features.shape[0]
        latent_vectors = []

        for i in range(batch_size):
            sample = features[i].detach().cpu().numpy()
            qc, ansatz = self._build_circuit(sample)
            # Bind parameters
            param_dict = {p: v for p, v in zip(ansatz.parameters, self.params)}
            qc = qc.bind_parameters(param_dict)
            result = self.simulator.run(qc).result()
            state = Statevector(result.get_statevector())
            amp = state.data
            # Convert to torch tensor (real, imag)
            amp_tensor = torch.tensor(amp, dtype=torch.complex64, device=self.device)
            latent_vectors.append(amp_tensor)

        latent = torch.stack(latent_vectors, dim=0)
        return latent

    def reconstruct(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct features from latent statevectors."""
        # Use the magnitude squared of amplitudes as probabilities
        probs = torch.abs(latent) ** 2
        probs = probs / probs.sum(dim=1, keepdim=True)
        recon = self.decoder(probs)
        return recon

class QuantumGraphNeuralNetwork(nn.Module):
    """Build a graph from latent quantum states using fidelity."""

    def __init__(self,
                 graph_threshold: float = 0.8,
                 graph_secondary: float | None = None,
                 graph_secondary_weight: float = 0.5) -> None:
        super().__init__()
        self.graph_threshold = graph_threshold
        self.graph_secondary = graph_secondary
        self.graph_secondary_weight = graph_secondary_weight

    def latent_to_graph(self, latent: torch.Tensor) -> nx.Graph:
        """Construct a weighted graph from latent statevectors."""
        num_samples = latent.shape[0]
        graph = nx.Graph()
        graph.add_nodes_from(range(num_samples))

        # Compute fidelity between all pairs
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                fid = torch.abs(torch.dot(latent[i], latent[j].conj())) ** 2
                fid = fid.item()
                if fid >= self.graph_threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif self.graph_secondary is not None and fid >= self.graph_secondary:
                    graph.add_edge(i, j, weight=self.graph_secondary_weight)
        return graph

    def forward(self, latent: torch.Tensor) -> nx.Graph:
        return self.latent_to_graph(latent)

__all__ = ["VariationalQuantumAutoEncoder", "QuantumGraphNeuralNetwork"]
