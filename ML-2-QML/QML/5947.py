import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np

Tensor = pnp.ndarray

# --------------------------------------------------------------------------- #
#  Random unitary generation and training data
# --------------------------------------------------------------------------- #

def _random_qubit_unitary(num_qubits: int) -> Tensor:
    """Return a random unitary matrix of size 2**num_qubits."""
    dim = 2 ** num_qubits
    mat = pnp.random.randn(dim, dim) + 1j * pnp.random.randn(dim, dim)
    u, _ = pnp.linalg.qr(mat)
    return u

def random_training_data(unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate random input states and their images under the target unitary."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        state = pnp.random.randn(dim) + 1j * pnp.random.randn(dim)
        state /= pnp.linalg.norm(state)
        target = unitary @ state
        dataset.append((state, target))
    return dataset

# --------------------------------------------------------------------------- #
#  Variational circuit architecture
# --------------------------------------------------------------------------- #

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random variational circuit architecture and training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training = random_training_data(target_unitary, samples)

    # Store parameters for each layer as a list of numpy arrays
    params: List[List[np.ndarray]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        layer_params: List[np.ndarray] = []
        for _ in range(num_inputs):
            # Each qubit receives a random rotation angle
            layer_params.append(pnp.random.randn())
        params.append(layer_params)

    return qnn_arch, params, training, target_unitary

# --------------------------------------------------------------------------- #
#  Forward propagation – apply one layer of the variational circuit
# --------------------------------------------------------------------------- #

def _apply_layer(state: Tensor, layer_params: List[np.ndarray], wires: Sequence[int]) -> Tensor:
    """Apply one layer of RY rotations followed by a CNOT chain."""
    dev = qml.device("default.qubit", wires=len(wires))
    @qml.qnode(dev)
    def circuit(inp: Tensor):
        qml.StatePrep(inp, wires=wires)
        for qubit, angle in enumerate(layer_params):
            qml.RY(angle, wires=wires[qubit])
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        return qml.state()
    return circuit(state)

def feedforward(
    qnn_arch: Sequence[int],
    params: List[List[np.ndarray]],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Forward pass for a batch of quantum samples."""
    stored_states: List[List[Tensor]] = []
    for state, _ in samples:
        layerwise: List[Tensor] = [state]
        current = state
        for layer in range(1, len(qnn_arch)):
            current = _apply_layer(current, params[layer], wires=range(qnn_arch[layer]))
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states

# --------------------------------------------------------------------------- #
#  Fidelity helpers – state overlap and graph construction
# --------------------------------------------------------------------------- #

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between pure states a and b."""
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(
    states: Sequence[Tensor],
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
#  Optional compilation to a Qiskit backend
# --------------------------------------------------------------------------- #

def compile_and_run(
    qnn_arch: Sequence[int],
    params: List[List[np.ndarray]],
    samples: Iterable[Tuple[Tensor, Tensor]],
    shots: int = 1024,
    backend_name: str = "qiskit.aer",
) -> List[Tensor]:
    """Compile the variational circuit to a Qiskit backend and execute."""
    import pennylane_qiskit as qiskit

    dev = qml.device(backend_name, wires=qnn_arch[-1], shots=shots)

    @qml.qnode(dev)
    def circuit(inp: Tensor, layer_params: List[List[np.ndarray]]):
        qml.StatePrep(inp, wires=range(qnn_arch[-1]))
        for layer in layer_params[1:]:
            for qubit, angle in enumerate(layer):
                qml.RY(angle, wires=qubit)
            for i in range(len(layer) - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.state()

    outputs: List[Tensor] = []
    for state, _ in samples:
        outputs.append(circuit(state, params))
    return outputs

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "compile_and_run",
]
