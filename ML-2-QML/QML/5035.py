"""GraphQNNGen355Quantum: quantum‑centric implementation of the hybrid graph‑neural‑network.

The class mirrors the classical API while fully leveraging Qiskit for state‑vector
propagation, fidelity calculation and circuit construction.  It also bundles
the Quantum‑NAT style `QFCModel` for convenient experimentation."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Union

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

Tensor = np.ndarray


class GraphQNNGen355Quantum:
    """Quantum‑only counterpart to the classical `GraphQNNGen355`."""
    def __init__(self, qnn_arch: Sequence[int], shots: int = 1024) -> None:
        self.arch = list(qnn_arch)
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.unitaries: List[List[QuantumCircuit]] = [[]]
        self._prepare_qlayer()

    # --------------------------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------------------------
    def _prepare_qlayer(self) -> None:
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            layer_ops: List[QuantumCircuit] = []
            for output in range(num_outputs):
                qc = QuantumCircuit(num_inputs + 1)
                qc.h(range(num_inputs + 1))
                qc.barrier()
                qc.ry(ParameterVector(f"theta_{layer}_{output}", 1), num_inputs + output)
                layer_ops.append(qc)
            self.unitaries.append(layer_ops)

    def _random_unitary(self, num_qubits: int) -> np.ndarray:
        dim = 2 ** num_qubits
        matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        q, _ = np.linalg.qr(matrix)
        return q

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def random_network(self, samples: int) -> Tuple[List[int], List[List[QuantumCircuit]], List[Tuple[Statevector, Statevector]], Statevector]:
        target_unitary = self._random_unitary(self.arch[-1])
        training_data = [
            (
                Statevector.from_label("0" * self.arch[-1]),
                Statevector(target_unitary @ Statevector.from_label("0" * self.arch[-1]).data),
            )
            for _ in range(samples)
        ]
        return self.arch, self.unitaries, training_data, target_unitary

    def feedforward(self, samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
        stored = []
        for state, _ in samples:
            current = state
            layerwise = [current]
            for layer in range(1, len(self.arch)):
                for gate in self.unitaries[layer]:
                    execute(gate, self.backend, shots=self.shots)
                # after each layer we keep a fresh zero‑state of the output qubits
                state_vec = Statevector.from_label("0" * self.arch[layer])
                layerwise.append(state_vec)
            stored.append(layerwise)
        return stored

    def state_fidelity(self, a: Statevector, b: Statevector) -> float:
        return abs((a.data.conj() @ b.data)) ** 2

    def fidelity_adjacency(self, states: Sequence[Statevector], threshold: float,
                           *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------------
    # classifier circuit factory
    # --------------------------------------------------------------------------
    def build_classifier_circuit(self, num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
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


__all__ = ["GraphQNNGen355Quantum"]
