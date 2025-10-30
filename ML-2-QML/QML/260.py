import networkx as nx
import qutip as qt
import scipy as sc
import pennylane as qml
import numpy as np
import itertools
from typing import Iterable, List, Sequence, Tuple

# --- Core QNN state propagation -------------------------------------------------

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
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
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
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
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
#   Extension: Variational PennyLane Circuits
# --------------------------------------------------------------------------- #

def _pennylane_variational_circuit(num_qubits: int, num_layers: int):
    """Return a PennyLane QNode that implements a layered RY–CNOT circuit."""
    dev = qml.device('default.qubit', wires=num_qubits)

    @qml.qnode(dev)
    def circuit(input_state: np.ndarray, params: np.ndarray):
        # Prepare the input state
        qml.QubitStateVector(input_state, wires=range(num_qubits))
        idx = 0
        for _ in range(num_layers):
            for q in range(num_qubits):
                qml.RY(params[idx], wires=q)
                idx += 1
            # Entangling layer
            for q in range(num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.state()
    return circuit

def random_variational_network(qnn_arch: Sequence[int], samples: int):
    """
    Construct a variational circuit with random parameters and training data.
    Returns (arch, params, circuit, training_data, target_unitary).
    """
    num_qubits = qnn_arch[0]
    num_layers = len(qnn_arch) - 1
    target_unitary = _random_qubit_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)
    params = np.random.randn(num_layers * num_qubits)
    circuit = _pennylane_variational_circuit(num_qubits, num_layers)
    return qnn_arch, params, circuit, training_data, target_unitary

def feedforward_variational(
    qnn_arch: Sequence[int],
    params: np.ndarray,
    circuit,
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[qt.Qobj]:
    """Run the variational circuit on each input state and return output states."""
    out_states: List[qt.Qobj] = []
    for input_state, _ in samples:
        vec = np.array(input_state.full()).flatten()
        out_vec = circuit(vec, params)
        out_state = qt.Qobj(out_vec.reshape((-1, 1)), dims=[input_state.dims[0], [1] * len(input_state.dims[0])])
        out_states.append(out_state)
    return out_states

def fidelity_loss(
    params: np.ndarray,
    qnn_arch: Sequence[int],
    circuit,
    training_data: List[Tuple[qt.Qobj, qt.Qobj]],
) -> float:
    """Mean loss defined as 1 - fidelity over the training set."""
    total = 0.0
    for input_state, target_state in training_data:
        vec = np.array(input_state.full()).flatten()
        out_vec = circuit(vec, params)
        out_state = qt.Qobj(out_vec.reshape((-1, 1)), dims=[input_state.dims[0], [1] * len(input_state.dims[0])])
        fid = state_fidelity(out_state, target_state)
        total += (1 - fid)
    return total / len(training_data)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "feedforward_variational",
    "random_variational_network",
    "fidelity_loss",
]
