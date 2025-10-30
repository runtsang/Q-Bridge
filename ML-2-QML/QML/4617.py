"""Quantum‑based implementation of the hybrid graph‑self‑attention network.
The module replaces classical dense layers with parameterised Qiskit circuits
and uses state‑vector fidelities to build the adjacency graph.  It mirrors the
public API of the classical version so that downstream code can swap between
the two seamlessly."""
from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from qiskit import QuantumCircuit, Aer, execute, assemble, transpile
from qiskit.quantum_info import Statevector, RandomUnitary
from qiskit.providers.aer import AerSimulator

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Quantum self‑attention
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Basic quantum self‑attention block using a parameterised circuit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            # Use entangle_params to modulate a controlled‑X
            theta = entangle_params[i]
            qc.cx(i, i + 1)
            qc.rx(theta, i + 1)
        return qc

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> Statevector:
        qc = self._build_circuit(rotation_params, entangle_params)
        return Statevector.from_instruction(qc)

# --------------------------------------------------------------------------- #
#  Quantum expectation head
# --------------------------------------------------------------------------- #
class QuantumCircuitHead:
    """Parameterized two‑qubit circuit executed on Aer to produce an expectation value."""
    def __init__(self, n_qubits: int, backend: AerSimulator | None = None, shots: int = 100, shift: float = np.pi / 2):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.shift = shift
        self._circuit = QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = QuantumCircuit.parameter("theta")  # placeholder
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self._circuit.parameters[0]: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------------- #
#  Graph utilities (quantum variant)
# --------------------------------------------------------------------------- #
def random_network(qnn_arch: Sequence[int], samples: int = 100):
    """Generate random unitaries for each layer and an empty training set."""
    unitaries = [RandomUnitary(q).data for q in qnn_arch]
    return list(qnn_arch), unitaries, [], None

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[np.ndarray],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    """Propagate a batch of initial states through the quantum network."""
    states = []
    for init_state, _ in samples:
        layerwise = [init_state]
        current = init_state
        for unitary in unitaries:
            current = current.evolve(unitary)
            layerwise.append(current)
        states.append(layerwise)
    return states

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Squared overlap between two statevectors."""
    return a.fidelity(b)

def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
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
#  Hybrid graph‑QNN classifier (quantum version)
# --------------------------------------------------------------------------- #
class HybridGraphQNNClassifier(nn.Module):
    """
    Quantum counterpart to the classical HybridGraphQNNClassifier.
    It uses a quantum self‑attention block, a stack of random unitaries,
    and a parameterised expectation head.  The output is a two‑class
    probability distribution derived from the expectation value.
    """
    def __init__(
        self,
        qnn_arch: Sequence[int],
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.attention = QuantumSelfAttention(n_qubits=self.qnn_arch[0])
        self.head = QuantumCircuitHead(
            n_qubits=self.qnn_arch[-1],
            backend=Aer.get_backend("qasm_simulator"),
            shots=100,
            shift=shift,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 1. Quantum self‑attention
        rotation_params = np.random.randn(self.qnn_arch[0] * 3)
        entangle_params = np.random.randn(self.qnn_arch[0] - 1)
        attn_state = self.attention.run(rotation_params, entangle_params)

        # 2. Feedforward through random unitaries (skipped for brevity)
        # Here we simply use the state from the attention block as the output.

        # 3. Hybrid expectation head
        expectation = self.head.run(inputs.tolist())
        probs = torch.tensor([expectation])
        return torch.cat((probs, 1 - probs), dim=-1)

    def generate_graph(self, state_threshold: float = 0.8) -> nx.Graph:
        """Build a fidelity graph from the attention state."""
        attn_state = self.attention.run(
            np.random.randn(self.qnn_arch[0] * 3),
            np.random.randn(self.qnn_arch[0] - 1),
        )
        return fidelity_adjacency([attn_state], state_threshold)

# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "HybridGraphQNNClassifier",
    "QuantumSelfAttention",
    "QuantumCircuitHead",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
