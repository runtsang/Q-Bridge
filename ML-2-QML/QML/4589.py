from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import torch
from typing import Sequence, Tuple, List
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.providers.aer import AerSimulator

# --------------------------------------------------------------------------- #
# Hybrid Graph‑QNN – quantum side
# --------------------------------------------------------------------------- #
class HybridGraphQNN:
    def __init__(self, arch: Sequence[int], gamma: float = 1.0):
        self.arch = list(arch)
        self.gamma = gamma
        self.sim = AerSimulator(method="statevector")

    # --------------------------------------------------------------------- #
    # Random unitary / state helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> Operator:
        dim = 2 ** num_qubits
        mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        mat, _ = np.linalg.qr(mat)
        return Operator(mat)

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> Statevector:
        dim = 2 ** num_qubits
        vec = np.random.randn(dim) + 1j * np.random.randn(dim)
        vec /= np.linalg.norm(vec)
        return Statevector(vec)

    @staticmethod
    def random_training_data(unitary: Operator, samples: int) -> List[Tuple[Statevector, Statevector]]:
        dataset: List[Tuple[Statevector, Statevector]] = []
        num_qubits = unitary.num_qubits
        for _ in range(samples):
            state = HybridGraphQNN._random_qubit_state(num_qubits)
            target = state.evolve(unitary)
            dataset.append((state, target))
        return dataset

    @staticmethod
    def random_network(arch: Sequence[int], samples: int) -> Tuple[List[int], List[List[Operator]], List[Tuple[Statevector, Statevector]], Operator]:
        target = HybridGraphQNN._random_qubit_unitary(arch[-1])
        training_data = HybridGraphQNN.random_training_data(target, samples)
        unitaries: List[List[Operator]] = [[]]
        for layer in range(1, len(arch)):
            num_inputs = arch[layer - 1]
            num_outputs = arch[layer]
            layer_ops: List[Operator] = []
            for _ in range(num_outputs):
                layer_ops.append(HybridGraphQNN._random_qubit_unitary(num_inputs + 1))
            unitaries.append(layer_ops)
        return list(arch), unitaries, training_data, target

    # --------------------------------------------------------------------- #
    # Forward propagation with partial‑trace
    # --------------------------------------------------------------------- #
    def _partial_trace(self, state: Statevector, keep: List[int]) -> Statevector:
        return state.ptrace(keep)

    def feedforward(self, sample: Statevector) -> List[Statevector]:
        states = [sample]
        current = sample
        for layer in range(1, len(self.arch)):
            # Apply a fresh random unitary and trace out the input registers
            unitary = self._random_qubit_unitary(self.arch[layer])
            current = current.evolve(unitary)
            current = self._partial_trace(current, list(range(self.arch[layer])))
            states.append(current)
        return states

    # --------------------------------------------------------------------- #
    # Fidelity and graph
    # --------------------------------------------------------------------- #
    def state_fidelity(self, a: Statevector, b: Statevector) -> float:
        return abs((a.data.conj().T @ b.data).item()) ** 2

    def fidelity_adjacency(self, states: Sequence[Statevector], threshold: float,
                           *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, j in itertools.combinations(range(len(states)), 2):
            fid = self.state_fidelity(states[i], states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------- #
    # Quantum kernel – encode states as rotations and compute overlap
    # --------------------------------------------------------------------- #
    def kernel_matrix(self, a: Sequence[Statevector], b: Sequence[Statevector]) -> np.ndarray:
        def encode(state: Statevector, sign: int = 1) -> QuantumCircuit:
            qc = QuantumCircuit(state.num_qubits)
            for idx, amp in enumerate(state.data):
                angle = sign * np.angle(amp)
                qc.ry(angle, idx)
            return qc

        gram = np.zeros((len(a), len(b)))
        for i, ax in enumerate(a):
            for j, by in enumerate(b):
                qc = QuantumCircuit(ax.num_qubits)
                qc.append(encode(ax), range(ax.num_qubits))
                qc.append(encode(by, sign=-1), range(by.num_qubits))
                state = Statevector(self.sim.run(qc).result().get_statevector(qc))
                gram[i, j] = abs(state.data[0]) ** 2
        return gram

__all__ = ["HybridGraphQNN"]
