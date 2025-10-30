"""Quantum regression and graph utilities.

This module combines the quantum regression circuit, a graph‑based
quantum neural network, and a classifier circuit.  All components
are built with Qiskit and its machine‑learning extensions.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, random_unitary
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
import networkx as nx
from typing import Iterable, List, Tuple, Sequence

# ----- Data generation ---------------------------------------------------------

def generate_superposition_data(num_qubits: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create states cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>."""
    zero = np.zeros(2 ** num_qubits, dtype=complex)
    zero[0] = 1.0
    one = np.zeros(2 ** num_qubits, dtype=complex)
    one[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_qubits), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * zero + np.exp(1j * phis[i]) * np.sin(thetas[i]) * one
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns quantum states and regression targets."""
    def __init__(self, samples: int, num_qubits: int):
        self.states, self.labels = generate_superposition_data(num_qubits, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return torch.tensor(self.states[idx], dtype=torch.cfloat), torch.tensor(self.labels[idx], dtype=torch.float32)

# ----- Quantum regression model -----------------------------------------------

class QuantumRegressionModel(nn.Module):
    """
    Variational regression model built on Qiskit’s EstimatorQNN.
    The circuit is a shallow Ry‑ansatz with a single observable.
    """
    def __init__(self, num_qubits: int, depth: int = 2):
        super().__init__()
        self.circuit, self.input_params, self.weight_params, self.observables = build_classifier_circuit(num_qubits, depth)
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the regression circuit on a batch of state vectors.
        """
        x = states.cpu().numpy()
        preds = self.qnn.predict(x)
        return torch.tensor(preds, dtype=torch.float32, device=states.device).squeeze(-1)

# ----- Graph‑based quantum neural network -------------------------------------

def random_unitary_matrix(num_qubits: int) -> np.ndarray:
    """Return a Haar‑random unitary as a NumPy array."""
    return random_unitary(2 ** num_qubits).data

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[List[np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Generate a random sequence of unitaries for each layer."""
    num_qubits = qnn_arch[0]
    unitaries: List[List[np.ndarray]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_f = qnn_arch[layer - 1]
        out_f = qnn_arch[layer]
        layer_ops: List[np.ndarray] = []
        for _ in range(out_f):
            U = random_unitary_matrix(in_f + 1)  # +1 for ancilla
            layer_ops.append(U)
        unitaries.append(layer_ops)

    target_unitary = random_unitary_matrix(qnn_arch[-1])
    training_data: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        state = np.random.randn(2 ** num_qubits) + 1j * np.random.randn(2 ** num_qubits)
        state /= np.linalg.norm(state)
        training_data.append((state, target_unitary @ state))
    return list(qnn_arch), unitaries, training_data, target_unitary

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[np.ndarray]], samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
    """Propagate quantum states through the random unitary network."""
    outputs: List[List[np.ndarray]] = []
    for state, _ in samples:
        layer_states = [state]
        current = state
        for layer, ops in enumerate(unitaries[1:], start=1):
            current = ops[0] @ current
            for op in ops[1:]:
                current = op @ current
            layer_states.append(current)
        outputs.append(layer_states)
    return outputs

def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted graph from the fidelity of pure states."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = np.abs(np.vdot(a, b)) ** 2
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# ----- Classifier circuit factory ---------------------------------------------

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    Returns the circuit, encoding parameters, weight parameters, and observables.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

# ------------------------------------------------------------------------------

__all__ = [
    "RegressionDataset",
    "QuantumRegressionModel",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "build_classifier_circuit",
    "generate_superposition_data",
]
