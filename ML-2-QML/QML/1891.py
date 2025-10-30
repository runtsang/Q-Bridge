"""Quantum‑only graph neural network variant using Qiskit.

The module defines:
  * `GraphQNN` – a pure‑quantum model that builds a variational ansatz from a
    graph architecture.
  * Helper functions mirroring the original interface: `random_network`,
    `feedforward`, `fidelity_adjacency`, `state_fidelity`.

The implementation focuses on a compact ansatz:
  * Input encoding with RY rotations.
  * `n_layers` repetitions of RX–RZ rotations plus a linear‑chain CNOT entanglement.
  * Parameter‑shift gradients for optimisation.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

# --------------------------------------------------------------------------- #
# 1.  Statevector helper
# --------------------------------------------------------------------------- #
def _statevector(qc: QuantumCircuit, backend=Aer.get_backend("statevector_simulator")) -> torch.Tensor:
    """Return the statevector as a complex torch tensor."""
    job = execute(qc, backend)
    statevector = job.result().get_statevector()
    return torch.tensor(statevector, dtype=torch.complex64)

# --------------------------------------------------------------------------- #
# 2.  Graph utilities
# --------------------------------------------------------------------------- #
def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float, *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = torch.dot(a, b).abs().item() ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the absolute squared overlap between two pure states."""
    return float((torch.dot(a, b).abs() ** 2).item())

# --------------------------------------------------------------------------- #
# 3.  Pure‑quantum model
# --------------------------------------------------------------------------- #
class GraphQNN:
    """Pure‑quantum graph neural network with variational ansatz."""
    def __init__(self, qnn_arch: Sequence[int]):
        self.arch = list(qnn_arch)
        self.num_qubits = self.arch[-1]
        self.n_layers = len(self.arch) - 1
        # Initialise parameters as a flat torch vector
        self.q_params = torch.randn(self.n_layers * self.num_qubits * 2, requires_grad=False)

    def _build_circuit(self, input_angles: torch.Tensor) -> QuantumCircuit:
        """Construct the variational circuit for a single example."""
        qc = QuantumCircuit(self.num_qubits)
        # Input encoding: RY(2·angle) per qubit
        for i in range(self.num_qubits):
            qc.ry(2 * input_angles[i].item(), i)
        # Variational layers
        for layer in range(self.n_layers):
            for q in range(self.num_qubits):
                idx = layer * self.num_qubits * 2 + q * 2
                theta = self.q_params[idx].item()
                phi = self.q_params[idx + 1].item()
                qc.rx(theta, q)
                qc.rz(phi, q)
            # Entanglement: linear chain
            for q in range(self.num_qubits - 1):
                qc.cx(q, q + 1)
        return qc

    def forward(self, input_angles: torch.Tensor) -> torch.Tensor:
        """Return the statevector for the given input angles."""
        qc = self._build_circuit(input_angles)
        return _statevector(qc)

    def train_step(self, data: Iterable[Tuple[torch.Tensor, torch.Tensor]], lr: float = 1e-3):
        """
        Simple parameter‑shift optimiser that updates `self.q_params` to minimise
        1 – fidelity between predicted and target states.
        """
        loss = 0.0
        grads = torch.zeros_like(self.q_params)
        for inp, tgt in data:
            pred = self.forward(inp)
            loss += (1 - abs(torch.dot(pred.conj(), tgt)) ** 2).item()
        loss /= len(data)

        # Parameter‑shift gradients
        for idx in range(len(self.q_params)):
            orig = self.q_params[idx].item()
            with torch.no_grad():
                self.q_params.data[idx] = orig + np.pi / 2
            loss_plus = 0.0
            for inp, tgt in data:
                pred = self.forward(inp)
                loss_plus += (1 - abs(torch.dot(pred.conj(), tgt)) ** 2).item()
            loss_plus /= len(data)

            with torch.no_grad():
                self.q_params.data[idx] = orig - np.pi / 2
            loss_minus = 0.0
            for inp, tgt in data:
                pred = self.forward(inp)
                loss_minus += (1 - abs(torch.dot(pred.conj(), tgt)) ** 2).item()
            loss_minus /= len(data)

            grads[idx] = (loss_plus - loss_minus) / 2
            with torch.no_grad():
                self.q_params.data[idx] = orig

        # Update parameters
        with torch.no_grad():
            self.q_params.data -= lr * grads

        return loss

# --------------------------------------------------------------------------- #
# 4.  Convenience functions
# --------------------------------------------------------------------------- #
def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random architecture, quantum parameters, and training data."""
    n_qubits = qnn_arch[-1]
    n_layers = len(qnn_arch) - 1
    q_params = torch.randn(n_layers * n_qubits * 2)

    training_data = []
    for _ in range(samples):
        # Random input angles
        inp_angles = torch.randn(n_qubits)
        # Build a random target unitary (different from the training parameters)
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.ry(2 * inp_angles[i].item(), i)
        for layer in range(n_layers):
            for q in range(n_qubits):
                theta = np.random.randn()
                phi = np.random.randn()
                qc.rx(theta, q)
                qc.rz(phi, q)
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
        target_state = _statevector(qc)
        training_data.append((inp_angles, target_state))

    return list(qnn_arch), q_params, training_data, None

def feedforward(qnn_arch: Sequence[int], q_params: torch.Tensor,
                samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
    """Evaluate the model on a batch of input angles."""
    model = GraphQNN(qnn_arch)
    model.q_params = q_params
    return [model.forward(inp) for inp, _ in samples]
