"""QuantumHybridEstimator – quantum side with variational QNN + fidelity graph."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import pennylane as qml
import pennylane.numpy as npq
import networkx as nx
import qiskit
import torch
from torch import nn

# ----------------------------------------------------------------------
# 1️⃣  Variational QNN layer (inspired by FCL and GraphQNN)
# ----------------------------------------------------------------------
class VariationalQNNLayer(nn.Module):
    """
    A torch‑module that evaluates a parameterised quantum circuit
    for each row of a batch.  It returns the expectation value of
    a fixed observable.
    """
    def __init__(
        self,
        num_qubits: int,
        observable: qml.operation.Operator,
        wires: Sequence[int] | None = None,
        dev: qml.Device | None = None,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.wires = wires or list(range(num_qubits))
        self.observable = observable
        self.device = dev or qml.device("default.qubit", wires=self.num_qubits)

        # Create a qnode that accepts a list of parameters
        self._qnode = qml.qnode(self._circuit, device=self.device)

    def _circuit(self, *params: float) -> float:
        """Simple variational circuit – one RY per qubit."""
        for w, p in zip(self.wires, params):
            qml.RY(p, wires=w)
        return qml.expval(self.observable)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # Convert each row to a list of parameters
        param_list = batch.cpu().numpy()
        # Evaluate the qnode for each parameter set
        results = [self._qnode(*params) for params in param_list]
        return torch.tensor(results, dtype=torch.float32, device=batch.device)


# ----------------------------------------------------------------------
# 2️⃣  Fidelity‑based graph utilities (inspired by GraphQNN)
# ----------------------------------------------------------------------
def state_fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Overlap squared of two pure state vectors."""
    a_norm = state_a / np.linalg.norm(state_a)
    b_norm = state_b / np.linalg.norm(state_b)
    return float(np.abs(np.vdot(a_norm, b_norm)) ** 2)


def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Construct a weighted graph where edges represent state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j, b in enumerate(states[i + 1 :], start=i + 1):
            fid = state_fidelity(a, b)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
    return G


# ----------------------------------------------------------------------
# 3️⃣  Quantum transformer block (inspired by QTransformerTorch)
# ----------------------------------------------------------------------
class QuantumFeedForward(nn.Module):
    """Feed‑forward block that optionally uses a small quantum circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, dev: qml.Device | None = None):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dev = dev
        if dev is not None:
            self.q_layer = VariationalQNNLayer(1, qml.PauliZ, wires=[0], dev=dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        if self.dev is not None:
            # Apply the quantum layer to each feature independently
            flat = x.view(-1)
            out = torch.tensor(
                [self.q_layer([p]).item() for p in flat.cpu().numpy()],
                dtype=torch.float32,
                device=x.device,
            ).view_as(x)
            x = out
        x = torch.relu(x)
        return self.linear2(x)


class QuantumTransformerBlock(nn.Module):
    """Single transformer block with classical attention and optional quantum feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dev: qml.Device | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Classical multi‑head attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, dev=dev)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ----------------------------------------------------------------------
# 4️⃣  Quantum circuit wrapper with shot noise (inspired by QTransformerTorch)
# ----------------------------------------------------------------------
class QuantumCircuitWithNoise:
    """
    Thin wrapper around a Qiskit circuit that automatically
    simulates shot noise when a number of shots is supplied.
    """
    def __init__(self, circuit: qiskit.QuantumCircuit, shots: int | None = None):
        self.circuit = circuit
        self.shots = shots

    def evaluate(self, parameters: Sequence[float]) -> np.ndarray:
        """Return expectation value of the circuit after binding parameters."""
        circ = self.circuit.copy()
        for param, val in zip(circ.parameters, parameters):
            circ.assign_parameters({param: val}, inplace=True)
        job = qiskit.execute(
            circ,
            qiskit.Aer.get_backend("qasm_simulator"),
            shots=self.shots or 1000,
        )
        result = job.result()
        counts = result.get_counts(circ)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])  # binary to int
        return np.sum(states * probs)


# ----------------------------------------------------------------------
# 5️⃣  Random network generator (graph‑like)
# ----------------------------------------------------------------------
def random_quantum_network(
    arch: Sequence[int], samples: int
) -> tuple[Sequence[int], List[List[torch.Tensor]], List[tuple[torch.Tensor, torch.Tensor]]]:
    """
    Generate a toy quantum neural network: a list of unitary layers
    and training data consisting of random input states and their
    transformed counterparts.
    """
    # Simple random unitaries using numpy
    unitaries = []
    for in_f, out_f in zip(arch[:-1], arch[1:]):
        dim = 2 ** in_f
        mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        u, _ = np.linalg.qr(mat)
        unitaries.append(torch.tensor(u, dtype=torch.cfloat))

    data = []
    for _ in range(samples):
        state = torch.randn(2 ** arch[0], dtype=torch.cfloat)
        state = state / torch.norm(state)
        target = torch.matmul(unitaries[-1], state)
        data.append((state, target))
    return arch, unitaries, data


__all__ = [
    "VariationalQNNLayer",
    "state_fidelity",
    "fidelity_adjacency",
    "QuantumFeedForward",
    "QuantumTransformerBlock",
    "QuantumCircuitWithNoise",
    "random_quantum_network",
]
