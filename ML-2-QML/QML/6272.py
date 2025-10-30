"""Hybrid quantum graph neural network with optional self‑attention circuits.

The quantum module mirrors the classical interface but replaces linear
layers with random unitaries and attention layers with Qiskit circuits
encapsulated in a lightweight wrapper.  The fidelity‑based graph
construction remains unchanged, allowing direct comparison between
classical and quantum embeddings."""
from __future__ import annotations

import itertools
import numpy as np
import qutip as qt
import networkx as nx
from typing import Iterable, List, Sequence, Tuple, Dict, Any
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit import Aer, execute

Tensor = qt.Qobj

# --------------------------------------------------------------------------- #
# Quantum self‑attention wrapper
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Qiskit‑based self‑attention circuit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        return circuit

    def run(self, state: qt.Qobj, rotation_params: np.ndarray, entangle_params: np.ndarray) -> qt.Qobj:
        circuit = self._build_circuit(rotation_params, entangle_params)
        sv = Statevector(state.data)
        sv = sv.evolve(circuit)
        return qt.Qobj(sv.data, dims=state.dims)


# --------------------------------------------------------------------------- #
# Core utilities
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.qr(matrix)[0]
    return qt.Qobj(unitary, dims=[[2] * num_qubits, [2] * num_qubits])


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amp = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amp /= np.linalg.norm(amp)
    return qt.Qobj(amp, dims=[[2] * num_qubits, [1] * num_qubits])


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(
    qnn_arch: Sequence[int],
    attention_cfg: Sequence[Dict[str, Any] | None],
    samples: int,
) -> Tuple[Sequence[int], List[Tensor | Dict[str, Any]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    """
    Build a hybrid quantum architecture.

    ``attention_cfg`` matches the classical counterpart: ``None`` for a
    linear unitary layer, a dictionary for a self‑attention block.
    """
    layers: List[qt.Qobj | Dict[str, Any]] = []
    for layer, cfg in zip(qnn_arch[:-1], attention_cfg):
        if cfg is None:
            layers.append(_random_qubit_unitary(layer))
        else:
            assert cfg["type"] == "attention"
            attn = QuantumSelfAttention(cfg["embed_dim"])
            rot = np.random.normal(size=3 * cfg["embed_dim"])
            ent = np.random.normal(size=cfg["embed_dim"] - 1)
            layers.append(
                {
                    "attention": attn,
                    "rot": rot,
                    "ent": ent,
                }
            )
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    return qnn_arch, layers, training_data, target_unitary


def _apply_unitary(state: qt.Qobj, unitary: qt.Qobj) -> qt.Qobj:
    return unitary * state


def _apply_attention(state: qt.Qobj, attn_dict: Dict[str, Any]) -> qt.Qobj:
    attn = attn_dict["attention"]
    return attn.run(state, attn_dict["rot"], attn_dict["ent"])


def feedforward(
    qnn_arch: Sequence[int],
    layers: Sequence[qt.Qobj | Dict[str, Any]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    stored: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        states = [sample]
        current = sample
        for layer in layers:
            if isinstance(layer, dict):
                current = _apply_attention(current, layer)
            else:
                current = _apply_unitary(current, layer)
            states.append(current)
        stored.append(states)
    return stored


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj], threshold: float,
    *, secondary: float | None = None, secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "QuantumSelfAttention",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "random_training_data",
]
