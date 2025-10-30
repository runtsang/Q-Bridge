"""GraphQNNHybrid – quantum implementation.

The class implements the same public API as the classical version but
uses qiskit variational circuits for each graph layer, random unitary
generators for training data, and a fidelity‑based graph construction.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit import Aer, execute

class GraphQNNHybrid:
    """Quantum‑based analogue of GraphQNNHybrid."""

    def __init__(self, qnn_arch: Sequence[int], backend: str = "qasm_simulator", shots: int = 1024) -> None:
        self.arch = list(qnn_arch)
        self.backend = Aer.get_backend(backend)
        self.shots = shots
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuits()

    def _build_circuits(self) -> Tuple[qiskit.QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """Create a layered ansatz with encoding and variational parameters."""
        encoding = ParameterVector("x", self.arch[0])
        weights = ParameterVector("theta", self.arch[0] * sum(1 for _ in self.arch[1:]))
        circuit = qiskit.QuantumCircuit(self.arch[0])
        for qubit, param in zip(range(self.arch[0]), encoding):
            circuit.rx(param, qubit)
        idx = 0
        for _ in range(len(self.arch) - 1):
            for qubit in range(self.arch[0]):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.arch[0] - 1):
                circuit.cz(qubit, qubit + 1)
        circuit.measure_all()
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.arch[0] - i - 1))
            for i in range(self.arch[0])
        ]
        return circuit, encoding, weights, observables

    def feedforward(self, samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
        """Execute the circuit on each sample and return the probability vectors."""
        stored_states: List[List[np.ndarray]] = []
        for sample, _ in samples:
            param_binds = [{self.encoding[i]: float(v) for i, v in enumerate(sample)}]
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
            result = job.result()
            counts = result.get_counts()
            state = np.zeros(2 ** self.arch[0])
            for bitstring, cnt in counts.items():
                idx = int(bitstring[::-1], 2)
                state[idx] = cnt / self.shots
            stored_states.append([state])
        return stored_states

    @staticmethod
    def random_training_data(target: qiskit.QuantumCircuit, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate synthetic data by applying the target circuit to random inputs."""
        dataset: List[Tuple[np.ndarray, np.ndarray]] = []
        num_qubits = target.num_qubits
        for _ in range(samples):
            inp = np.random.rand(num_qubits)
            param_binds = [{target.parameters[i]: float(v) for i, v in enumerate(inp)}]
            job = execute(target, target.backend, shots=1, parameter_binds=param_binds)
            result = job.result().get_counts()
            out_state = np.zeros(2 ** num_qubits)
            for bs, cnt in result.items():
                out_state[int(bs[::-1], 2)] = cnt
            dataset.append((inp, out_state))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Create a random circuit and training data."""
        num_qubits = qnn_arch[0]
        circuit = qiskit.QuantumCircuit(num_qubits)
        for qubit in range(num_qubits):
            circuit.rx(np.random.rand() * 2 * np.pi, qubit)
        for _ in range(sum(1 for _ in qnn_arch[1:])):
            for qubit in range(num_qubits):
                circuit.ry(np.random.rand() * 2 * np.pi, qubit)
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        circuit.measure_all()
        training_data = GraphQNNHybrid.random_training_data(circuit, samples)
        return list(qnn_arch), circuit, training_data, circuit

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Overlap of two probability distributions."""
        return float(np.abs(np.vdot(a, b)) ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[np.ndarray], threshold: float,
        *, secondary: float | None = None,
        secondary_weight: float = 0.5
    ) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, sa), (j, sb) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(sa, sb)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int, depth: int
    ) -> Tuple[qiskit.QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """Return the same tuple as the classical build but with quantum objects."""
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)
        circuit = qiskit.QuantumCircuit(num_qubits)
        for qubit, param in zip(range(num_qubits), encoding):
            circuit.rx(param, qubit)
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return circuit, list(encoding), list(weights), observables


__all__ = ["GraphQNNHybrid"]
