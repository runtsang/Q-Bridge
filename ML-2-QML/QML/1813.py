from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector


class GraphQNN:
    """Hybrid variational graph neural network using Qiskit."""

    def __init__(self, arch: Sequence[int], circuits: Sequence[Sequence[QuantumCircuit]]):
        self.arch = list(arch)
        self.circuits = circuits

    def feedforward(
        self, samples: Iterable[Tuple[Statevector, Statevector]]
    ) -> list[list[Statevector]]:
        results: list[list[Statevector]] = []
        for sample, _ in samples:
            layerwise: list[Statevector] = [sample]
            current = sample
            for layer in range(1, len(self.arch)):
                for gate in self.circuits[layer]:
                    current = Statevector(gate) @ current
                layerwise.append(current)
            results.append(layerwise)
        return results

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        return abs(np.vdot(a.data, b.data)) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_training_data(
        target_circuit: QuantumCircuit, samples: int
    ) -> list[tuple[Statevector, Statevector]]:
        data: list[tuple[Statevector, Statevector]] = []
        backend = Aer.get_backend("statevector_simulator")
        for _ in range(samples):
            # random input state
            init = Statevector.from_label("0" * target_circuit.num_qubits)
            # random unitary before measurement
            random_circ = RealAmplitudes(target_circuit.num_qubits, reps=1)
            job = execute(random_circ, backend, shots=1)
            rand_state = Statevector(job.result().get_statevector())
            # target transformation
            target_state = Statevector(target_circuit) @ init
            data.append((rand_state, target_state))
        return data

    @staticmethod
    def random_network(arch: list[int], samples: int):
        circuits: list[list[QuantumCircuit]] = [[]]
        for layer in range(1, len(arch)):
            layer_ops: list[QuantumCircuit] = []
            for _ in range(arch[layer]):
                op = RealAmplitudes(arch[layer - 1], reps=1)
                layer_ops.append(op)
            circuits.append(layer_ops)
        target_circuit = RealAmplitudes(arch[-1], reps=1)
        training_data = GraphQNN.random_training_data(target_circuit, samples)
        return GraphQNN(arch, circuits), training_data, target_circuit

    @staticmethod
    def sample_from_circuit(circuit: QuantumCircuit, shots: int = 1024) -> np.ndarray:
        backend = Aer.get_backend("qasm_simulator")
        job = execute(circuit, backend, shots=shots)
        counts = job.result().get_counts()
        return np.array([int(k, 2) for k in counts.keys()])

    @staticmethod
    def graph_optimizer(graph: nx.Graph, max_cluster_size: int = 5) -> dict[int, int]:
        clusters: dict[int, int] = {}
        cluster_id = 0
        visited: set[int] = set()
        for node in graph.nodes():
            if node in visited:
                continue
            cluster: set[int] = {node}
            for neighbor in graph.neighbors(node):
                if neighbor not in visited and len(cluster) < max_cluster_size:
                    cluster.add(neighbor)
            for n in cluster:
                visited.add(n)
                clusters[n] = cluster_id
            cluster_id += 1
        return clusters
