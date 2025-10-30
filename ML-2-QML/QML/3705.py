"""HybridGraphAutoencoder – quantum implementation.

Provides a quantum feed‑forward pipeline, fidelity‑based graph construction,
and a quantum autoencoder built with Qiskit.  It mirrors the classical
module while exposing quantum‑specific utilities."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

def _tensored_zero(num_qubits: int) -> np.ndarray:
    zero = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
    zero[0, 0] = 1.0
    return zero


def _random_unitary(num_qubits: int) -> np.ndarray:
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    return q


def _random_state(num_qubits: int) -> np.ndarray:
    dim = 2 ** num_qubits
    amp = np.random.randn(dim) + 1j * np.random.randn(dim)
    amp /= np.linalg.norm(amp)
    return amp


def random_training_data(
    unitary: np.ndarray, samples: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        state = _random_state(int(np.log2(unitary.shape[0])))
        dataset.append((state, unitary @ state))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random target unitary and per‑layer unitaries."""
    target_unitary = _random_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[np.ndarray]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_q = qnn_arch[layer - 1]
        out_q = qnn_arch[layer]
        layer_ops: List[np.ndarray] = []
        for _ in range(out_q):
            op = _random_unitary(in_q + 1)  # +1 for auxiliary qubit
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, training_data, target_unitary


def _partial_trace(state: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    """Partial trace over qubits not in `keep`."""
    num_qubits = int(np.log2(state.shape[0]))
    reshaped = state.reshape([2] * num_qubits + [2] * num_qubits)
    for qubit in sorted(set(range(num_qubits)) - set(keep), reverse=True):
        reshaped = np.trace(reshaped, axis1=qubit, axis2=qubit + num_qubits)
    return reshaped.reshape(-1)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[np.ndarray]],
    layer: int,
    input_state: np.ndarray,
) -> np.ndarray:
    in_q = qnn_arch[layer - 1]
    out_q = qnn_arch[layer]
    state = np.kron(input_state, _tensored_zero(out_q))
    unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        unitary = gate @ unitary
    new_state = unitary @ state
    keep = list(range(in_q))
    return _partial_trace(new_state, keep)


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[np.ndarray]],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Quantum feed‑forward propagation through the network."""
    stored = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        stored.append(layerwise)
    return stored


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap squared between two pure states."""
    return abs(np.vdot(a, b)) ** 2


def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def build_autoencoder_qnn(num_latent: int, num_trash: int) -> SamplerQNN:
    """Construct a quantum autoencoder using a swap‑test style ansatz."""
    algorithm_globals.random_seed = 42
    sampler = StatevectorSampler()

    def ansatz(num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=5)

    def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        qc.barrier()
        auxiliary = num_latent + 2 * num_trash
        for i in range(num_trash):
            qc.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
        qc.h(auxiliary)
        qc.measure(auxiliary, cr[0])
        return qc

    qc = auto_encoder_circuit(num_latent, num_trash)
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )
    return qnn


def train_qnn(
    qnn: SamplerQNN,
    training_data: List[Tuple[np.ndarray, np.ndarray]],
    epochs: int = 100,
    optimizer_cls=COBYLA,
) -> List[float]:
    """Train a quantum neural network using a classical optimizer."""
    opt = optimizer_cls()
    losses: List[float] = []
    for _ in range(epochs):
        loss = qnn.fit(training_data, opt)
        losses.append(loss)
    return losses


__all__ = [
    "build_autoencoder_qnn",
    "train_qnn",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
]
