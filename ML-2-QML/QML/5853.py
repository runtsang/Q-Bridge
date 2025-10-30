"""GraphQNN__gen252.py – Quantum graph neural network with variational auto‑encoder.

The module extends the original quantum utilities by:
* A variational circuit that encodes input states into a latent sub‑space
* A training loop that optimises a fidelity‑based loss using PennyLane.
* A ``train`` method that returns the fidelity graph of latent states.
* An ``evaluate`` helper that builds the graph from a test set.

The classical interface (feedforward, fidelity_adjacency, etc.) is preserved.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
import qutip as qt

# --------------------------------------------------------------------------- #
#  Core helpers – unchanged from the original seed
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector

def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.qr(matrix)[0]
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes /= np.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]):
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise: List[qt.Qobj] = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
) -> nx.Graph:
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
#  Variational auto‑encoder – quantum circuit
# --------------------------------------------------------------------------- #
class QuantumAutoEncoder:
    """A simple variational circuit that encodes an input state into a latent sub‑space
    and decodes it back to the full Hilbert space.  The latent dimension is
    represented by the first ``latent_dim`` qubits.
    """

    def __init__(self, num_qubits: int, latent_dim: int, layers: int = 2):
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.layers = layers
        # Parameters: (layers, num_qubits, 3) – one RX,RZ,RZ per qubit
        self.params = np.random.uniform(0, 2 * np.pi, size=(layers, num_qubits, 3))

    def circuit(self, state: qt.Qobj, params: np.ndarray) -> np.ndarray:
        """Return the output state produced by the variational circuit."""
        dev = qml.device("default.qubit", wires=self.num_qubits)
        @qml.qnode(dev, interface="autograd")
        def circuit_tape():
            qml.QubitStateVector(state.full(), wires=range(self.num_qubits))
            for l in range(self.layers):
                for qubit in range(self.num_qubits):
                    qml.RX(params[l, qubit, 0], wires=qubit)
                    qml.RY(params[l, qubit, 1], wires=qubit)
                    qml.RZ(params[l, qubit, 2], wires=qubit)
                for qubit in range(self.num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            return qml.state()
        return circuit_tape()

    def latent_state(self, state: qt.Qobj, params: np.ndarray) -> qt.Qobj:
        """Return the reduced state on the first ``latent_dim`` qubits."""
        full_np = self.circuit(state, params)
        full_qobj = qt.Qobj(full_np)
        full_qobj.dims = [[2]*self.num_qubits, [1]*self.num_qubits]
        return qt.partial_trace(full_qobj, list(range(self.latent_dim)))

# --------------------------------------------------------------------------- #
#  Training routine
# --------------------------------------------------------------------------- #
def train(
    num_qubits: int,
    latent_dim: int,
    training_data: List[Tuple[qt.Qobj, qt.Qobj]],
    epochs: int = 200,
    lr: float = 0.01,
) -> nx.Graph:
    """Train the variational auto‑encoder to reproduce the target states.

    The loss is 1 – fidelity between the circuit output and the target state.
    After training a fidelity graph is built from the latent states of the
    training set.
    """
    encoder = QuantumAutoEncoder(num_qubits, latent_dim)
    opt = qml.AdamOptimizer(stepsize=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inp, target in training_data:
            def cost(params):
                out_state = encoder.circuit(inp, params)
                return 1.0 - qml.state_fidelity(out_state, target.full())

            grads = qml.grad(cost)(encoder.params)
            encoder.params = opt.step(grads, encoder.params)
            epoch_loss += cost(encoder.params)
        # Optional: print or log epoch_loss / len(training_data)
    # Build latent graph
    latent_states = [encoder.latent_state(inp, encoder.params) for inp, _ in training_data]
    return fidelity_adjacency(latent_states, threshold=0.95)

# --------------------------------------------------------------------------- #
#  Evaluation helper
# --------------------------------------------------------------------------- #
def evaluate(
    num_qubits: int,
    latent_dim: int,
    test_data: List[Tuple[qt.Qobj, qt.Qobj]],
    threshold: float = 0.95,
) -> nx.Graph:
    """Return a fidelity graph of latent representations on a test set."""
    encoder = QuantumAutoEncoder(num_qubits, latent_dim)
    # Random parameters – no training
    latent_states = [encoder.latent_state(inp, encoder.params) for inp, _ in test_data]
    return fidelity_adjacency(latent_states, threshold=threshold)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train",
    "evaluate",
    "QuantumAutoEncoder",
]
