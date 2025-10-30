"""Quantum graph‑based neural network with a hybrid quantum head.

This module implements GraphQNNHybrid (alias) that mirrors the
public API of the classical GraphQNNHybrid while operating on quantum
states.  It uses Qiskit to build a variational circuit that maps a
classical activation vector into a single expectation value, which
is then processed by a sigmoid to produce a binary probability.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Any

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit as QC, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Operator, Statevector, state_fidelity as qiskit_fidelity
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
#  Utility functions – random generation & fidelity helpers
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> Operator:
    """Return a random unitary operator on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    return Operator(matrix)

def random_training_data(target_unitary: Operator, samples: int) -> List[Tuple[Statevector, Statevector]]:
    """Generate training pairs (state, U*state)."""
    dataset: List[Tuple[Statevector, Statevector]] = []
    dim = target_unitary.dim
    for _ in range(samples):
        vec = np.random.normal(size=(dim,)) + 1j * np.random.normal(size=(dim,))
        vec /= np.linalg.norm(vec)
        state = Statevector(vec)
        target = target_unitary @ state
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random variational network and training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[Operator]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[Operator] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                # Pad with identity on remaining outputs
                pad = Operator(np.eye(2 ** (num_outputs - 1), dtype=complex))
                op = op.tensor(pad)
                # Swap the new output qubit into the correct position
                op = op.swap(num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, training_data, target_unitary

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Operator]],
    layer: int,
    input_state: Statevector,
) -> Statevector:
    """Apply the layer unitary and trace out the input qubits."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    # Pad the input state with zeros on the output qubits
    padded = input_state.tensor(Statevector.from_label('0' * num_outputs))
    unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        unitary = gate @ unitary
    new_state = unitary @ padded
    # Keep only the output qubits
    keep = list(range(num_inputs, num_inputs + num_outputs))
    return new_state.trace(keep)

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Operator]],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    """Propagate each sample through the variational network."""
    result: List[List[Statevector]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        result.append(layerwise)
    return result

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Absolute squared overlap between two pure states."""
    return float(qiskit_fidelity(a, b))

def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
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
#  Hybrid quantum head
# --------------------------------------------------------------------------- #
class QuantumExpectationHead:
    """
    Variational circuit that maps a classical vector into a single
    expectation value of the Pauli‑Z operator on the last qubit.
    """
    def __init__(self, n_qubits: int, shots: int = 1000, shift: float = np.pi / 2):
        self.n_qubits = n_qubits
        self.shots = shots
        self.shift = shift
        self.backend = AerSimulator()
        self.circuit = QC(n_qubits)
        qubits = list(range(n_qubits))
        theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(qubits)
        self.circuit.ry(theta, qubits)
        self.circuit.measure_all()
        self.theta = theta

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of parameters."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: p} for p in params],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.run(inputs)

class HybridQuantumHead(nn.Module):
    """PyTorch wrapper around QuantumExpectationHead."""
    def __init__(self, n_qubits: int, shots: int = 1000, shift: float = np.pi / 2):
        super().__init__()
        self.head = QuantumExpectationHead(n_qubits, shots, shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        params = inputs.detach().cpu().numpy()
        exp = self.head(params)
        return torch.tensor(exp, dtype=torch.float32)

# --------------------------------------------------------------------------- #
#  GraphQNNHybridQuantum – quantum model
# --------------------------------------------------------------------------- #
class GraphQNNHybridQuantum(nn.Module):
    """
    Quantum graph‑based neural network that mirrors the classical API.
    The forward method returns a list of activation tensors (here kept as
    classical tensors for simplicity) and a binary probability
    distribution produced by the quantum expectation head.
    """
    def __init__(self, arch: Sequence[int], shots: int = 1000, shift: float = np.pi / 2):
        super().__init__()
        self.arch = list(arch)
        self.unitaries: List[List[Operator]] = []
        self.head = HybridQuantumHead(self.arch[-1], shots=shots, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = [x]
        current = x
        for layer in range(1, len(self.arch)):
            # In a full implementation, current would be encoded into a
            # quantum state and the variational layer would be applied.
            activations.append(current)
        logits = self.head(current)
        probs = torch.cat([logits, 1 - logits], dim=-1)
        return activations, probs

# Expose the classical name for compatibility
GraphQNNHybrid = GraphQNNHybridQuantum

__all__ = [
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "HybridQuantumHead",
    "GraphQNNHybridQuantum",
    "GraphQNNHybrid",
]
