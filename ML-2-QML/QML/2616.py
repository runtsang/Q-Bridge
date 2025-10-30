import numpy as np
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable, Optional

# --------------------------------------------------------------------------- #
# Core quantum utilities
# --------------------------------------------------------------------------- #

def _tensored_identity(num_qubits: int) -> np.ndarray:
    """Full‑system identity with proper shape."""
    dim = 2 ** num_qubits
    return np.eye(dim, dtype=complex)

def _tensored_zero(num_qubits: int) -> np.ndarray:
    """Zero state |0…0⟩ as a column vector."""
    dim = 2 ** num_qubits
    vec = np.zeros((dim, 1), dtype=complex)
    vec[0, 0] = 1.0
    return vec

def _swap_registers(state: np.ndarray, source: int, target: int) -> np.ndarray:
    """Swap two qubits in a state vector by permuting basis indices."""
    if source == target:
        return state
    dim = state.shape[0]
    perm = np.arange(dim)
    for idx in range(dim):
        bits = list(bin(idx)[2:].zfill(int(np.log2(dim))))
        bits[source], bits[target] = bits[target], bits[source]
        new_idx = int(''.join(bits), 2)
        perm[idx] = new_idx
    return state[perm, :]

def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Sample a Haar‑random unitary on 2^n dimensions."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    return q

def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Sample a random pure state on 2^n dimensions."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim, 1) + 1j * np.random.randn(dim, 1)
    vec /= np.linalg.norm(vec)
    return vec

def random_training_data(target: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate (input, target) pairs where target = U |ψ⟩."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    dim = target.shape[0]
    for _ in range(samples):
        psi = _random_qubit_state(int(np.log2(dim)))
        dataset.append((psi, target @ psi))
    return dataset

def random_network(arch: Sequence[int], samples: int):
    """Build a random layered unitary network and a training set."""
    unitaries: List[List[np.ndarray]] = [[]]
    for layer in range(1, len(arch)):
        in_q = arch[layer - 1]
        out_q = arch[layer]
        layer_ops: List[np.ndarray] = []
        for _ in range(out_q):
            op = _random_qubit_unitary(in_q + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)
    target_unitary = _random_qubit_unitary(arch[-1])
    training = random_training_data(target_unitary, samples)
    return list(arch), unitaries, training, target_unitary

def _partial_trace(state: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    """Compute the partial trace over all qubits not in ``keep``."""
    dim = state.shape[0]
    num_qubits = int(np.log2(dim))
    keep = sorted(keep)
    trace_out = [q for q in range(num_qubits) if q not in keep]
    reshaped = state.reshape([2]*num_qubits*2)
    for q in reversed(trace_out):
        reshaped = np.trace(reshaped, axis1=q, axis2=q+num_qubits)
    return reshaped.reshape([2]*len(keep), [2]*len(keep))

def _layer_channel(arch: Sequence[int], unitaries: Sequence[Sequence[np.ndarray]],
                   layer: int, input_state: np.ndarray) -> np.ndarray:
    """Apply a single layer of the quantum network."""
    in_q = arch[layer - 1]
    out_q = arch[layer]
    ancilla = _tensored_zero(out_q)
    state = np.kron(input_state, ancilla)
    U = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        U = gate @ U
    state = U @ state
    return _partial_trace(state, range(out_q))

def feedforward(arch: Sequence[int], unitaries: Sequence[Sequence[np.ndarray]],
                data: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
    """Collect layer‑wise states for each sample."""
    all_states: List[List[np.ndarray]] = []
    for inp, _ in data:
        states = [inp]
        curr = inp
        for layer in range(1, len(arch)):
            curr = _layer_channel(arch, unitaries, layer, curr)
            states.append(curr)
        all_states.append(states)
    return all_states

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared absolute inner product of two pure states."""
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float,
                       *, secondary: Optional[float] = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
# Quantum quanvolution primitives using PennyLane
# --------------------------------------------------------------------------- #

import pennylane as qml

def _qnode_for_patch(num_qubits: int, params: Sequence[float]):
    """Create a PennyLane QNode that applies a random 2‑qubit circuit."""
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev, interface="numpy")
    def circuit(psi, params):
        for i, val in enumerate(psi):
            qml.RY(val, wires=i)
        for i in range(num_qubits):
            qml.RY(params[i], wires=i)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
    return circuit

class QuantumQuanvolutionFilter:
    """Apply a quantum kernel to 2×2 image patches using PennyLane."""
    def __init__(self, n_qubits: int = 4, n_params: int = 4):
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.params = np.random.randn(n_params)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass on a batch of grayscale images (N, H, W)."""
        N, H, W = x.shape
        patches: List[np.ndarray] = []
        for r in range(0, H, 2):
            for c in range(0, W, 2):
                patch = x[:, r:r+2, c:c+2].reshape(N, -1)
                out = []
                for sample in patch:
                    circuit = _qnode_for_patch(self.n_qubits, self.params)
                    out.append(circuit(sample))
                patches.append(np.array(out))
        return np.concatenate(patches, axis=1)

class QuantumQuanvolutionClassifier:
    """Hybrid classifier: quantum filter + classical linear head."""
    def __init__(self, n_classes: int = 10):
        self.filter = QuantumQuanvolutionFilter()
        self.n_classes = n_classes

    def __call__(self, x: np.ndarray) -> np.ndarray:
        features = self.filter(x)
        W = np.random.randn(features.shape[1], self.n_classes)
        logits = features @ W
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return np.log(probs)

# --------------------------------------------------------------------------- #
# Graph‑based hybrid wrapper (quantum)
# --------------------------------------------------------------------------- #

class GraphQNNHybridQML:
    """Quantum graph neural network with fidelity‑based graph construction."""
    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.unitaries, self.training, self.target = random_network(arch, samples=0)

    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        states = [x]
        for layer in range(1, len(self.arch)):
            x = _layer_channel(self.arch, self.unitaries, layer, x)
            states.append(x)
        return states

    def fidelity_graph(self, states: Sequence[np.ndarray], threshold: float,
                       *, secondary: Optional[float] = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                   secondary_weight=secondary_weight)

__all__ = [
    "GraphQNNHybridQML",
    "QuantumQuanvolutionFilter",
    "QuantumQuanvolutionClassifier",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
