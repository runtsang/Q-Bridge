"""
Hybrid Graph Neural Network for classical training with optional Qiskit quantum layers.

The module provides:
* `GraphQNNHybrid` – a `torch.nn.Module` that can operate purely classically,
  or inject a Qiskit variational circuit at any selected layer.
* Helper functions to generate random networks, synthetic training data,
  deterministic feed‑forward propagation and fidelity‑based adjacency graphs.
"""

from __future__ import annotations

import itertools
from typing import List, Tuple, Sequence, Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

# Qiskit imports – used only when a quantum layer is active
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector

Tensor = torch.Tensor
Device = torch.device


# --------------------------------------------------------------------------- #
# Classical utilities – deterministic feed‑forward
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix with a fixed seed for reproducibility."""
    torch.manual_seed(0)
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic data for a regression task."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Return architecture, weight list, training data and target weight."""
    weights: List[Tensor] = [
        _random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
    ]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Deterministic forward pass through a classical weight matrix chain."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        layer_out = [features]
        for w in weights:
            layer_out.append(torch.tanh(w @ layer_out[-1]))
        activations.append(layer_out)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two classical vectors."""
    an = a / (torch.norm(a) + 1e-12)
    bn = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(an, bn).item() ** 2)


def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                       *, secondary: Optional[float] = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph where edges reflect pairwise fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Quantum utilities – variational circuit per node
# --------------------------------------------------------------------------- #
def _qc_variational(node_vec: Tensor, n_qubits: int, params: List[float]) -> Tensor:
    """
    Map a classical node vector to a quantum state, run a small
    parameter‑tied circuit, and return the expectation values of a
    Pauli‑Z operator for each qubit.  The circuit is executed on the
    Aer state‑vector simulator for speed.
    """
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(qr, cr)

    # Encode the node vector as rotation angles
    for idx, val in enumerate(node_vec.tolist()):
        qc.ry(val, idx)

    # Parameterised Ry gates for each qubit
    for idx in range(n_qubits):
        par = Parameter(f"θ{idx}")
        qc.ry(par, idx)

    # Use state‑vector simulator to obtain expectation values
    backend = AerSimulator(method="statevector")
    qc = transpile(qc, backend=backend)
    param_dict = {f"θ{idx}": p for idx, p in enumerate(params)}
    qc = qc.bind_parameters(param_dict)

    job = backend.run(qc)
    result = job.result()
    statevector = result.get_statevector(qc)

    # Compute expectation of Z on each qubit
    exp_z = []
    for i in range(n_qubits):
        # Construct Pauli‑Z operator on qubit i
        z_op = np.eye(2)
        z_op[0, 0] = 1
        z_op[1, 1] = -1
        # Tensor product with identity for other qubits
        op = 1
        for j in range(n_qubits):
            op = np.kron(op, z_op if j == i else np.eye(2))
        exp_val = np.vdot(statevector, op @ statevector).real
        exp_z.append(exp_val)
    return torch.tensor(exp_z, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Hybrid Graph Neural Network
# --------------------------------------------------------------------------- #
class GraphQNNHybrid(nn.Module):
    """
    Hybrid graph neural network that can inject quantum‑derived features
    at any layer of the classical feed‑forward chain.

    Parameters
    ----------
    *qnn_arch : int
        The network architecture – each entry is a number of nodes in
        each layer.
    quantum_layers : Optional[List[int]]
        Layer indices (0‑based) that should be processed through the
        quantum circuit.  If empty, the output of all layers is only
        classical.
    n_qubits : int
        Number of quantum bits used in the quantum sub‑module; defaults
        to 2.
    device : torch.device
        Device for the model parameters.
    """

    def __init__(self,
                 *qnn_arch: int,
                 quantum_layers: Optional[List[int]] = None,
                 n_qubits: int = 2,
                 device: Device = torch.device("cpu")) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.quantum_layers = quantum_layers or []
        self.n_qubits = n_qubits
        self.device = device

        # Classical weight matrices
        self.weights = nn.ParameterList(
            nn.Parameter(torch.randn(out, in_).to(device))
            for in_, out in zip(self.qnn_arch[:-1], self.qnn_arch[1:])
        )

        # Map layer index to quantum parameter index
        self.q_param_index = {layer: idx for idx, layer in enumerate(self.quantum_layers)}

        # Quantum parameters – one per qubit for each quantum layer
        self.q_params = nn.ParameterList(
            nn.Parameter(torch.randn(self.n_qubits).to(device))
            for _ in self.quantum_layers
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.  Returns the final layer output.
        """
        activations = [x]
        for idx, w in enumerate(self.weights):
            if idx in self.quantum_layers:
                q_idx = self.q_param_index[idx]
                # Quantum route: compute quantum features and project to output dim
                q_features = _qc_variational(x, self.n_qubits, self.q_params[q_idx].tolist())
                out = F.linear(q_features, w.t(), bias=None)
            else:
                # Classical route
                out = torch.tanh(F.linear(x, w.t(), bias=None))
            activations.append(out)
            x = out
        return activations[-1]

    def fidelity_graph(self,
                       activations: List[Tensor],
                       threshold: float,
                       *, secondary: Optional[float] = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        """
        Construct a fidelity‑based adjacency graph from the last layer activations.
        """
        states = [act[-1] for act in activations]
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
