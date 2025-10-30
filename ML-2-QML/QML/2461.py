"""Graph‑based quantum neural network classifier (quantum backend)."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple, Optional

import networkx as nx
import qutip as qt
import scipy as sc
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.optimizers import SPSA

Qobj = qt.Qobj

# ----------------------------------------------------------------------
# Helper utilities (adapted from original QNN)
# ----------------------------------------------------------------------
def _init_tensored_id(num_qubits: int) -> Qobj:
    """Identity operator on a register of `num_qubits` qubits."""
    I = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    I.dims = [dims.copy(), dims.copy()]
    return I

def _init_tensored_zero(num_qubits: int) -> Qobj:
    """Zero projector on a register of `num_qubits` qubits."""
    proj = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    proj.dims = [dims.copy(), dims.copy()]
    return proj

def _swap_registers(op: Qobj, src: int, tgt: int) -> Qobj:
    """Permute qubit indices in a tensor product operator."""
    if src == tgt:
        return op
    order = list(range(len(op.dims[0])))
    order[src], order[tgt] = order[tgt], order[src]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> Qobj:
    """Generate a random unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    U = sc.linalg.orth(mat)
    qobj = qt.Qobj(U)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> Qobj:
    """Sample a random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    vec = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    vec /= sc.linalg.norm(vec)
    qobj = qt.Qobj(vec)
    qobj.dims = [[2] * num_qubits, [1] * num_qubits]
    return qobj

def random_training_data(unitary: Qobj, samples: int) -> List[Tuple[Qobj, Qobj]]:
    """Generate state‑label pairs by applying `unitary` to random inputs."""
    data: List[Tuple[Qobj, Qobj]] = []
    n = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(n)
        data.append((state, unitary * state))
    return data

def random_network(qnn_arch: List[int], samples: int):
    """Return architecture, list of unitary layers, training data, and target unitary."""
    target = _random_qubit_unitary(qnn_arch[-1])
    training = random_training_data(target, samples)

    layers: List[List[Qobj]] = [[]]
    for layer_idx in range(1, len(qnn_arch)):
        inp = qnn_arch[layer_idx - 1]
        outp = qnn_arch[layer_idx]
        ops: List[Qobj] = []
        for out in range(outp):
            op = _random_qubit_unitary(inp + 1)
            if outp > 1:
                op = qt.tensor(_random_qubit_unitary(inp + 1), _init_tensored_id(outp - 1))
                op = _swap_registers(op, inp, inp + out)
            ops.append(op)
        layers.append(ops)

    return list(qnn_arch), layers, training, target

def _partial_trace_keep(state: Qobj, keep: Sequence[int]) -> Qobj:
    """Partial trace over all qubits except those in `keep`."""
    if len(keep) == len(state.dims[0]):
        return state
    return state.ptrace(list(keep))

def _partial_trace_remove(state: Qobj, remove: Sequence[int]) -> Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Qobj]], layer: int, input_state: Qobj) -> Qobj:
    """Apply a single layer of the QNN to `input_state`."""
    inp = qnn_arch[layer - 1]
    outp = qnn_arch[layer]
    state = qt.tensor(input_state, _init_tensored_zero(outp))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(inp))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Qobj]], samples: Iterable[Tuple[Qobj, Qobj]]) -> List[List[Qobj]]:
    """Compute the state at each layer for every sample."""
    all_states: List[List[Qobj]] = []
    for sample, _ in samples:
        states = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            states.append(current)
        all_states.append(states)
    return all_states

def _state_fidelity(a: Qobj, b: Qobj) -> float:
    """Squared overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2

def _fidelity_graph(states: Sequence[Qobj], threshold: float, *, secondary: Optional[float] = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# ----------------------------------------------------------------------
# Quantum classifier builder (adapted)
# ----------------------------------------------------------------------
def _build_quantum_classifier(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Return a parameterised ansatz and metadata."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for q, param in zip(range(num_qubits), encoding):
        qc.ry(param, q)  # encoding with Ry

    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.rz(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------
class GraphQNNClassifier:
    """Unified quantum graph‑based QNN and node‑level classifier."""
    def __init__(self, qnn_arch: Sequence[int], classifier_depth: int = 2, backend: str ='statevector_simulator'):
        self.arch = list(qnn_arch)
        self.arch, self.unitaries, self.training_data, self.target_unitary = random_network(self.arch, samples=200)
        self.backend = Aer.get_backend(backend)
        self.classifier_circuit, self.enc, self.wts, self.obs = _build_quantum_classifier(self.arch[-1], classifier_depth)
        self.optimizer = SPSA(maxiter=200, disp=False)
        self.loss_history: List[float] = []

        # Create classification labels (random for demo)
        self.classifier_data: List[Tuple[Qobj, int]] = []
        for state, _ in self.training_data:
            label = int(sc.random.rand() > 0.5)
            self.classifier_data.append((state, label))

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------
    def graph_from_states(self, states: Sequence[Qobj], threshold: float, secondary: Optional[float] = None) -> nx.Graph:
        """Build adjacency graph from state fidelities."""
        return _fidelity_graph(states, threshold, secondary=secondary)

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------
    def forward(self, samples: Iterable[Tuple[Qobj, Qobj]]) -> List[List[Qobj]]:
        """Layer‑wise states for each sample."""
        return feedforward(self.arch, self.unitaries, samples)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_classifier(self, epochs: int = 50, lr: float = 0.01):
        """Variational training of the classifier circuit."""
        params = list(self.classifier_circuit.parameters)
        for _ in range(epochs):
            loss, _ = self.optimizer.optimize(lambda p: self._circuit_loss(p), len(params), params)
            self.loss_history.append(loss)

    def _circuit_loss(self, params) -> float:
        """Compute expectation‑value loss over training set."""
        self.classifier_circuit.assign_parameters(params)
        total = 0.0
        for state, label in self.classifier_data:
            job = execute(self.classifier_circuit, self.backend, initial_state=state.to_array())
            result = job.result()
            sv = result.get_statevector()
            sv_qobj = qt.Qobj(sv)
            exp = (sv_qobj.dag() * qt.sigmaz() * sv_qobj)[0, 0].real
            total += (1 - exp) if label == 1 else (1 + exp)
        return total / len(self.classifier_data)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, state: Qobj) -> int:
        """Return class label for a single input state."""
        job = execute(self.classifier_circuit, self.backend, initial_state=state.to_array())
        result = job.result()
        sv = result.get_statevector()
        sv_qobj = qt.Qobj(sv)
        exp = (sv_qobj.dag() * qt.sigmaz() * sv_qobj)[0, 0].real
        return int(exp > 0)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def generate_random_samples(self, samples: int) -> List[Tuple[Qobj, Qobj]]:
        """Generate synthetic data for a given number of samples."""
        return random_training_data(self.target_unitary, samples)

__all__ = ["GraphQNNClassifier"]
