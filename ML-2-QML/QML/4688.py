"""Quantum‑centric counterparts to the classical utilities defined above.

The quantum implementations use Qiskit, Qiskit Machine Learning, Qutip and
SciPy to provide the same API signatures while performing genuine quantum
operations.  Each function or class is designed to be drop‑in for code that
expects the seed modules.

Key components:

* :func:`build_classifier_circuit` – constructs a parameterised variational
  ansatz with data‑encoding and interaction layers.
* :class:`SamplerQNN` – a Qiskit‑based sampler network that returns a
  probability distribution over two classes.
* :class:`GraphQNNUtility` – utilities that generate random unitary networks,
  perform feed‑forward evolution of quantum states and build a fidelity graph.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Sequence
import itertools
import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler
import qutip as qt
import scipy as sc


# --------------------------------------------------------------------------- #
# 1. Quantum classifier circuit
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a variational ansatz with data‑encoding and interaction layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (also the dimension of the input encoding).
    depth : int
        Number of interaction layers (each contains an RY rotation per qubit
        followed by a chain of CZ gates).

    Returns
    -------
    circuit : QuantumCircuit
        Parameterised circuit ready for simulation.
    encoding : Iterable[Parameter]
        Parameters used for data encoding (one RX per qubit).
    weights : Iterable[Parameter]
        Parameters for the variational rotations.
    observables : List[SparsePauliOp]
        Z operators on each qubit – used as measurement observables.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
# 2. Quantum sampler network
# --------------------------------------------------------------------------- #
class SamplerQNN:
    """Wraps Qiskit’s SamplerQNN to expose a torch‑style API."""

    def __init__(self, num_qubits: int = 2) -> None:
        self.num_qubits = num_qubits
        self._build_circuit()

    def _build_circuit(self) -> None:
        inputs = ParameterVector("input", self.num_qubits)
        weights = ParameterVector("weight", 4)

        qc = QuantumCircuit(self.num_qubits)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        self.circuit = qc
        self.input_params = inputs
        self.weight_params = weights

        sampler = Sampler()
        self.sampler_qnn = QiskitSamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the sampler for a batch of input vectors.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, 2) – raw input features.

        Returns
        -------
        probs : np.ndarray
            Shape (batch, 2) – probability distribution over the two classes.
        """
        return self.sampler_qnn.predict(inputs)


def SamplerQNN_factory(num_qubits: int = 2) -> SamplerQNN:
    """Factory matching the seed signature."""
    return SamplerQNN(num_qubits=num_qubits)


# --------------------------------------------------------------------------- #
# 3. Graph‑based quantum utilities
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    identity.dims = [[2] * num_qubits, [2] * num_qubits]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    projector.dims = [[2] * num_qubits, [1] * num_qubits]
    return projector


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    qobj.dims = [[2] * num_qubits, [2] * num_qubits]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amps = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amps /= sc.linalg.norm(amps)
    state = qt.Qobj(amps)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate input–output pairs for a target unitary."""
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Create a random unitary network and training data."""
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
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    """Apply a single layer of the unitary network."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Evolve each sample through the network and collect all intermediate states."""
    all_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise: List[qt.Qobj] = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        all_states.append(layerwise)
    return all_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from pairwise fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "build_classifier_circuit",
    "SamplerQNN",
    "SamplerQNN_factory",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
