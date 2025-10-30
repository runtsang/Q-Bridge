"""Quantum graph neural network that uses a fidelity‑attention graph to
guide controlled interactions between qubits.  The module extends the
original GraphQNN QML seed by incorporating a Qiskit‑based
self‑attention circuit to produce edge weights that modulate the
variational circuit."""
from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torchquantum as tq
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

# Import helper utilities from the seed modules
from GraphQNN import _random_qubit_unitary as _rand_unitary
from GraphQNN import _random_qubit_state as _rand_state
from GraphQNN import random_training_data as _rand_train
from GraphQNN import fidelity_adjacency as _fid_adj
from SelfAttention import SelfAttention as _QSA_cls


class QuantumSelfAttention:
    """Minimal Qiskit self‑attention circuit producing a probability
    distribution over qubit pairs.  The measurement counts are mapped
    to a symmetric adjacency matrix."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits)
        self.cr = ClassicalRegister(n_qubits)
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rot_params: np.ndarray, ent_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rot_params[3 * i], i)
            circuit.ry(rot_params[3 * i + 1], i)
            circuit.rz(rot_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(ent_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rot_params: np.ndarray, ent_params: np.ndarray, shots: int = 1024) -> np.ndarray:
        circuit = self._build_circuit(rot_params, ent_params)
        job = execute(circuit, self.backend, shots=shots)
        counts = job.result().get_counts(circuit)
        # Convert counts to a symmetric adjacency matrix
        matrix = np.zeros((self.n_qubits, self.n_qubits))
        for bitstring, count in counts.items():
            bits = [int(b) for b in bitstring[::-1]]
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    if bits[i] == bits[j]:
                        matrix[i, j] += count
                        matrix[j, i] += count
        # Normalize
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix


class GraphQNNHybridQ(tq.QuantumModule):
    """Variational quantum graph neural network guided by a fidelity‑attention graph."""
    def __init__(self, qnn_arch: Sequence[int], adjacency: nx.Graph):
        super().__init__()
        self.arch = list(qnn_arch)
        self.adj = adjacency
        self.n_qubits = sum(qnn_arch)
        self.qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=1, device="cpu")

        # Build layers: each layer is a list of unitaries acting on a subset
        self.layers: List[List[tq.QuantumModule]] = []
        offset = 0
        for layer_idx in range(1, len(qnn_arch)):
            in_wires = list(range(offset, offset + qnn_arch[layer_idx - 1]))
            out_wires = list(range(offset + qnn_arch[layer_idx - 1],
                                    offset + qnn_arch[layer_idx]))
            offset += qnn_arch[layer_idx]
            layer_ops: List[tq.QuantumModule] = []

            # Random layer on input + output qubits
            rand_layer = tq.RandomLayer(n_ops=20, wires=in_wires + out_wires)
            layer_ops.append(rand_layer)

            # Controlled interactions guided by adjacency
            for src in in_wires:
                for dst in out_wires:
                    if self.adj.has_edge(src, dst):
                        weight = self.adj[src][dst]["weight"]
                        # Use a controlled‑RX with angle proportional to weight
                        crx = tq.CRX(weight * np.pi, wires=[src, dst])
                        layer_ops.append(crx)

            self.layers.append(layer_ops)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Propagate a batch of quantum states through the graph‑aware circuit."""
        bsz = state_batch.shape[0]
        self.qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz,
                                     device=state_batch.device)
        # Initialize state: encode as computational basis states
        self.qdev.set_state(state_batch)
        for layer_ops in self.layers:
            for op in layer_ops:
                op(self.qdev)
        # Measure all qubits in Z basis and return expectation values
        return self.qdev.expectation(tq.PauliZ)

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return tq.losses.mse(preds, targets)


def random_network(qnn_arch: List[int], samples: int):
    """Generate a random graph‑aware quantum circuit and training data."""
    # Random unitary for the final layer (used as training target)
    target_unitary = _rand_unitary(qnn_arch[-1])
    # Random training states and labels
    training_data = _rand_train(target_unitary, samples)

    # Build a random graph based on fidelity of random states
    states = [_rand_state(qnn_arch[-1]) for _ in range(samples)]
    adjacency = _fid_adj(states, threshold=0.7)

    return qnn_arch, adjacency, training_data, target_unitary


def random_training_data(unitary: tq.Qobj, samples: int) -> List[Tuple[tq.Qobj, tq.Qobj]]:
    """Generate training data from a target unitary."""
    data = []
    for _ in range(samples):
        state = _rand_state(unitary.dims[0][0])
        data.append((state, unitary * state))
    return data


__all__ = [
    "GraphQNNHybridQ",
    "random_network",
    "random_training_data",
    "feedforward",
    "fidelity_adjacency",
]
