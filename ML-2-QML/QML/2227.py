from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
import numpy as np
import networkx as nx
import qutip as qt
import scipy as sc
import itertools
from typing import List, Tuple

__all__ = ["UnifiedQuantumGraphLayer"]

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
    amps = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amps /= sc.linalg.norm(amps)
    state = qt.Qobj(amps)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        st = _random_qubit_state(num_qubits)
        dataset.append((st, unitary * st))
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

def _partial_trace_keep(state: qt.Qobj, keep: List[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qt.Qobj, remove: List[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: List[int], unitaries: List[List[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch: List[int], unitaries: List[List[qt.Qobj]], samples: List[Tuple[qt.Qobj, qt.Qobj]]):
    stored = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        stored.append(layerwise)
    return stored

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: List[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

class UnifiedQuantumGraphLayer:
    """
    Quantum implementation of a graph‑structured neural network.
    The architecture is defined by a list of integers describing the
    number of nodes per layer.  The circuit contains a parameterised
    Ry rotation on each qubit for every layer, optionally entangled
    according to a fidelity‑based adjacency graph.
    """

    def __init__(self, arch: List[int], shots: int = 1024):
        self.arch = arch
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self._build_circuit()

    def _build_circuit(self):
        n_qubits = self.arch[0]
        self.circuit = QuantumCircuit(n_qubits)
        self.params = {}
        for layer in range(1, len(self.arch)):
            for q in range(n_qubits):
                theta = Parameter(f"theta_{layer}_{q}")
                self.params[f"theta_{layer}_{q}"] = theta
                self.circuit.ry(theta, q)
            if layer < len(self.arch) - 1:
                self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas: List[List[float]]) -> np.ndarray:
        """
        Execute the circuit with a list of theta matrices.
        ``thetas`` must have shape (num_layers-1, n_qubits).
        """
        param_bindings = {}
        for layer in range(1, len(self.arch)):
            for q in range(self.arch[0]):
                param_bindings[f"theta_{layer}_{q}"] = thetas[layer-1][q]
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=[param_bindings])
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])
