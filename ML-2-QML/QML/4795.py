"""Quantum kernel + graph neural network utilities using TorchQuantum, Qiskit, and qutip.

This module mirrors the public API of the classical implementation while
offering quantum‑centric functionality.  It defines a `QuantumKernelGraphNet`
class that bundles a TorchQuantum kernel, a Qiskit fully‑connected layer,
and a qutip‑based graph neural network.  Each component is optional and can
be toggled via the constructor.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import networkx as nx
import qiskit
import qutip as qt
import scipy as sc

# --------------------------------------------------------------------------- #
# 1. Quantum kernel utilities (TorchQuantum)
# --------------------------------------------------------------------------- #
try:
    import torchquantum as tq
    from torchquantum.functional import func_name_dict
except Exception:
    raise ImportError("TorchQuantum is required for the quantum kernel implementation")

class KernalAnsatz(tq.QuantumModule):
    """Quantum RBF‑like kernel built from a list of gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            torchquantum.functional.func_name_dict[info["func"]](q_device,
                                                               wires=info["wires"],
                                                               params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            torchquantum.functional.func_name_dict[info["func"]](q_device,
                                                               wires=info["wires"],
                                                               params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the ansatz on input and returns overlap."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = KernalAnsatz(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def quantum_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
# 2. Quantum fully‑connected layer (Qiskit)
# --------------------------------------------------------------------------- #
class QuantumFCL:
    """A parameterised quantum circuit that acts as a fully‑connected layer."""
    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        circ.h(range(self.n_qubits))
        circ.barrier()
        circ.ry(theta, range(self.n_qubits))
        circ.measure_all()
        return circ

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.circuit.parameters[0]: theta} for theta in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])


# --------------------------------------------------------------------------- #
# 3. Quantum graph neural network utilities (qutip)
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

def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: list[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[qt.Qobj] = []
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

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                   layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(),
                                 range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                samples: Iterable[tuple[qt.Qobj, qt.Qobj]]):
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
    """Return the absolute squared overlap between pure states `a` and `b`."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
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
# 4. Composite quantum class
# --------------------------------------------------------------------------- #
class QuantumKernelGraphNet:
    """
    Quantum‑centric wrapper that bundles:
      - a TorchQuantum‑based kernel
      - a Qiskit‑based fully‑connected layer
      - a qutip‑based graph neural network
    The API mirrors the classical version for ease of comparison.
    """
    def __init__(self,
                 n_wires: int = 4,
                 n_qubits_fcl: int = 1,
                 shots_fcl: int = 1024,
                 qnn_arch: Sequence[int] | None = None,
                 samples: int = 10):
        self.n_wires = n_wires
        self.kernel = Kernel(n_wires=n_wires)
        self.fcl = QuantumFCL(n_qubits=n_qubits_fcl, shots=shots_fcl)
        self.qnn_arch = qnn_arch
        self.samples = samples
        if qnn_arch is not None:
            self.qnn_arch, self.unitaries, self.training_data, self.target_unitary = random_network(list(qnn_arch), samples)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        return self.fcl.run(thetas)

    def feedforward(self, samples: Iterable[tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        return feedforward(self.qnn_arch, self.unitaries, samples)

    def fidelity_graph(self, states: Sequence[qt.Qobj], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "KernalAnsatz",
    "Kernel",
    "quantum_kernel_matrix",
    "QuantumFCL",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "QuantumKernelGraphNet",
]
