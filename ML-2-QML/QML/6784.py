import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import networkx as nx
import itertools
from typing import Iterable, List, Sequence, Tuple

class QuantumSelfAttention:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits)
        self.cr = ClassicalRegister(n_qubits)
        self.backend = Aer.get_backend('qasm_simulator')

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def generate_mask(self, num_heads: int, shots: int = 1024) -> np.ndarray:
        rot = np.random.uniform(0, np.pi, size=3 * self.n_qubits)
        ent = np.random.uniform(0, np.pi, size=self.n_qubits - 1)
        qc = self._build_circuit(rot, ent)
        job = execute(qc, self.backend, shots=shots)
        counts = job.result().get_counts(qc)
        probs = np.zeros(2 ** self.n_qubits, dtype=np.float32)
        for bitstring, c in counts.items():
            idx = int(bitstring, 2)
            probs[idx] = c
        probs /= probs.sum()
        head_probs = np.zeros(num_heads, dtype=np.float32)
        for i, p in enumerate(probs):
            head = i % num_heads
            head_probs[head] += p
        return head_probs / head_probs.sum()

class GraphQNN:
    def __init__(self, qnn_arch: Sequence[int]):
        self.qnn_arch = list(qnn_arch)
        self.unitaries = self._random_network()

    def _random_unitary(self, dim: int) -> np.ndarray:
        mat = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        q, _ = np.linalg.qr(mat)
        return q

    def _random_network(self) -> List[List[np.ndarray]]:
        unitaries: List[List[np.ndarray]] = [[]]
        for layer in range(1, len(self.qnn_arch)):
            in_f = self.qnn_arch[layer - 1]
            out_f = self.qnn_arch[layer]
            layer_ops: List[np.ndarray] = []
            for _ in range(out_f):
                dim = in_f + 1
                layer_ops.append(self._random_unitary(dim))
            unitaries.append(layer_ops)
        return unitaries

    def feedforward(self, samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
        stored_states: List[List[np.ndarray]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current = sample
            for layer in range(1, len(self.qnn_arch)):
                ops = self.unitaries[layer]
                state = np.concatenate([current, np.zeros((self.qnn_arch[layer], 1), dtype=complex)], axis=0)
                for op in ops:
                    state = op @ state
                current = state[:self.qnn_arch[layer]]
                layerwise.append(current)
            stored_states.append(layerwise)
        return stored_states

    def state_fidelity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = a / (np.linalg.norm(a) + 1e-12)
        b_norm = b / (np.linalg.norm(b) + 1e-12)
        return float(np.abs(np.vdot(a_norm, b_norm)) ** 2)

    def fidelity_adjacency(self, states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph
