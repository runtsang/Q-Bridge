"""Quantum self‑attention module that uses a variational circuit.

The class HybridSelfAttentionQNN implements:
- Building a variational circuit from a fidelity‑based graph.
- Executing the circuit on a simulator or real backend.
- Returning a probability distribution over attention heads.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
import qutip as qt
import scipy as sc

Qobj = qt.Qobj


# ----------------------------------------------------------------------------- #
#  Quantum utilities – adapted from GraphQNN.py
# ----------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: Qobj, source: int, target: int) -> Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: Qobj, samples: int) -> List[Tuple[Qobj, Qobj]]:
    dataset: List[Tuple[Qobj, Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: Qobj, keep: Sequence[int]) -> Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: Qobj, remove: Sequence[int]) -> Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Qobj]], layer: int, input_state: Qobj) -> Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Qobj]], samples: Iterable[Tuple[Qobj, Qobj]]) -> List[List[Qobj]]:
    stored_states: List[List[Qobj]] = []
    for sample, _ in samples:
        layerwise: List[Qobj] = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: Qobj, b: Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
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


# ----------------------------------------------------------------------------- #
#  HybridSelfAttentionQNN – quantum implementation
# ----------------------------------------------------------------------------- #

class HybridSelfAttentionQNN:
    """Quantum self‑attention network that uses a variational circuit.

    Parameters
    ----------
    qnn_arch : List[int]
        Architecture of the quantum graph network; the first element
        corresponds to the input qubits and the second to the next layer.
    backend : qiskit.providers.Provider or None, optional
        Quantum backend to execute the circuit.  If ``None`` a local
        Aer simulator is used.
    """

    def __init__(self, qnn_arch: List[int], backend=None):
        self.qnn_arch = qnn_arch
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

        # Build a random network and training data for demonstration
        _, self.unitaries, self.training_data, self.target_unitary = random_network(
            qnn_arch, samples=10
        )

        # Construct a fidelity‑based graph from the random states
        states = [sample[0] for sample in self.training_data]
        self.graph = fidelity_adjacency(states, threshold=0.8, secondary=0.6)

    def _build_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """Construct a parameterised variational circuit guided by the graph."""
        n_qubits = self.qnn_arch[0]
        qr = QuantumRegister(n_qubits, "q")
        cr = ClassicalRegister(n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Apply parameterised single‑qubit rotations
        for i in range(n_qubits):
            circuit.rx(params[3 * i], qr[i])
            circuit.ry(params[3 * i + 1], qr[i])
            circuit.rz(params[3 * i + 2], qr[i])

        # Entangle qubits according to the fidelity graph
        for (i, j, w) in self.graph.edges(data="weight"):
            circuit.cx(qr[i], qr[j])

        circuit.measure(qr, cr)
        return circuit

    def run(self, params: np.ndarray, shots: int = 1024) -> dict:
        """Execute the circuit and return measurement counts."""
        circuit = self._build_circuit(params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

    def quantum_forward(self, params: np.ndarray, shots: int = 1024) -> np.ndarray:
        """Return a probability distribution over attention heads."""
        counts = self.run(params, shots)
        total = sum(counts.values())
        probs = np.array([counts.get(bin(i)[2:].zfill(len(self.qnn_arch)), 0) for i in range(2 ** len(self.qnn_arch))])
        return probs / total if total > 0 else probs

    def train_step(self, params: np.ndarray, lr: float = 0.01) -> np.ndarray:
        """A simple gradient‑free update (e.g. COBYLA) for demonstration."""
        from scipy.optimize import minimize

        def loss(p):
            probs = self.quantum_forward(p)
            # Example loss: maximise probability of the all‑zero state
            return -probs[0]

        res = minimize(loss, params, method="COBYLA", options={"maxiter": 50})
        return res.x

    def feedforward(
        self,
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[Qobj]],
        samples: Iterable[Tuple[Qobj, Qobj]],
    ) -> List[List[Qobj]]:
        """Run the quantum feed‑forward network (for comparison)."""
        return feedforward(qnn_arch, unitaries, samples)

    def random_network(self, qnn_arch: Sequence[int], samples: int):
        """Generate a random network and training data."""
        return random_network(qnn_arch, samples)

    def random_training_data(self, unitary: Qobj, samples: int):
        """Generate random training data for a given unitary."""
        return random_training_data(unitary, samples)

    def state_fidelity(self, a: Qobj, b: Qobj) -> float:
        """Return fidelity between two quantum states."""
        return state_fidelity(a, b)

    def __repr__(self) -> str:
        return f"HybridSelfAttentionQNN(qnn_arch={self.qnn_arch})"
