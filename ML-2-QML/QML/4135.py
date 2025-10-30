"""Hybrid graph neural network with quantum LSTM and quantum auto‑encoder.

The quantum implementation mirrors :class:`HybridGraphQLSTMNet` from the
classic module but replaces the auto‑encoder and the LSTM with
variational quantum circuits.  The auto‑encoder prepares a quantum state
from the raw node features using RX rotations; a single CX gate is
added to introduce entanglement.  The statevector is sampled with
``StatevectorSampler`` and the Pauli‑Z expectation values of each qubit
are used as a real latent representation.  The LSTM is the
torchquantum‑based cell from the QLSTM seed.  Fidelity‑based adjacency
graphs are constructed from the sampled statevectors.

Key components
---------------
* :class:`QuantumAutoencoder` – prepares a quantum statevector from
  classical data and extracts a real latent vector.
* :class:`QLSTM` – quantum‑enhanced LSTM cell.
* :func:`fidelity_adjacency` – builds a weighted graph from state
  fidelities.
* :class:`HybridGraphQLSTMNet` – the hybrid quantum architecture.
"""

from __future__ import annotations

import itertools
import numpy as np
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler

# --------------------------------------------------------------------------- #
# Quantum auto‑encoder (Qiskit)
# --------------------------------------------------------------------------- #

class QuantumAutoencoder(nn.Module):
    """Variational quantum auto‑encoder using Qiskit.

    For each input vector the circuit prepares a state by applying RX
    rotations with the input values as angles.  A single CX gate is
    added to introduce entanglement when more than one qubit is used.
    The statevector is sampled with :class:`StatevectorSampler` and the
    Pauli‑Z expectation values of each qubit are returned as a real
    latent vector.
    """

    def __init__(self, num_latent: int) -> None:
        super().__init__()
        self.num_latent = num_latent
        self.sampler = StatevectorSampler()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of classical vectors into quantum statevectors.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (N, num_latent) – each row is a set of angles
            for the RX gates.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N, num_latent) containing the Pauli‑Z
            expectation values of each qubit.
        """
        batch_size = inputs.size(0)
        latent_vectors: List[np.ndarray] = []
        for idx in range(batch_size):
            angles = inputs[idx].cpu().numpy()
            qr = QuantumRegister(self.num_latent)
            qc = QuantumCircuit(qr)
            for i, angle in enumerate(angles):
                qc.rx(angle, qr[i])
            if self.num_latent > 1:
                qc.cx(qr[0], qr[1])
            result = self.sampler.run(qc).result()
            state = result.statevector
            # Compute Pauli‑Z expectation for each qubit
            exp_vals = np.zeros(self.num_latent, dtype=np.float32)
            dim = 2 ** self.num_latent
            for basis_state, amp in enumerate(state):
                prob = abs(amp) ** 2
                for q in range(self.num_latent):
                    bit = (basis_state >> q) & 1
                    exp_vals[q] += (1.0 if bit == 0 else -1.0) * prob
            latent_vectors.append(exp_vals)
        return torch.tensor(latent_vectors, dtype=torch.float32, device=inputs.device)


# --------------------------------------------------------------------------- #
# Quantum LSTM (from QLSTM seed)
# --------------------------------------------------------------------------- #

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell based on torchquantum."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs: List[torch.Tensor] = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
# Fidelity utilities (quantum version)
# --------------------------------------------------------------------------- #

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Quantum state fidelity for pure states represented as real vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Hybrid Quantum Graph‑LSTM auto‑encoder
# --------------------------------------------------------------------------- #

class HybridGraphQLSTMNet(nn.Module):
    """Hybrid graph neural network that uses a quantum auto‑encoder and
    a quantum LSTM to process graph node features.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw node features.
    hidden_dim : int
        Hidden size of the quantum LSTM cell.
    latent_dim : int
        Number of qubits used by the quantum auto‑encoder.
    n_qubits : int
        Number of qubits used by the quantum LSTM cell.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_qubits: int,
    ) -> None:
        super().__init__()
        self.autoencoder = QuantumAutoencoder(num_latent=latent_dim)
        self.lstm = QLSTM(input_dim=latent_dim, hidden_dim=hidden_dim, n_qubits=n_qubits)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass for the hybrid quantum architecture.

        Parameters
        ----------
        node_features : torch.Tensor
            Tensor of shape (N, D) where N is the number of nodes.
        adjacency : torch.Tensor
            Adjacency matrix of shape (N, N).

        Returns
        -------
        torch.Tensor
            Reconstructed node features of shape (N, D).
        """
        # 1. Encode node features into quantum latent vectors
        latent = self.autoencoder(node_features)  # (N, latent_dim)

        # 2. Treat each node as a timestep and feed into the quantum LSTM
        seq = latent.unsqueeze(1)  # (N, 1, latent_dim)
        lstm_out, _ = self.lstm(seq)  # (N, 1, hidden_dim)
        lstm_out = lstm_out.squeeze(1)  # (N, hidden_dim)

        # 3. Decode back to the original feature space
        recon = self.output_layer(lstm_out)  # (N, D)
        return recon

    def build_fidelity_graph(
        self,
        node_features: torch.Tensor,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Construct a weighted graph based on quantum state fidelities."""
        latent = self.autoencoder(node_features)
        states = [l.cpu() for l in latent]
        return fidelity_adjacency(
            states, threshold, secondary=secondary, secondary_weight=secondary_weight
        )
