"""QuantumKernelMethodGen207 – quantum implementation.

The quantum build mirrors the classical API but replaces kernel evaluation,
graph propagation, convolution and fully‑connected layers with quantum‑native
counterparts.  The class can be dropped into the same training pipelines
without modification.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import itertools
import networkx as nx
import numpy as np
import qiskit
import qutip as qt
import scipy as sc
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# 1. Quantum kernel utilities
# --------------------------------------------------------------------------- #

class KernalAnsatz(tq.QuantumModule):
    """Data‑encoding ansatz used for kernel evaluation."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: Tensor, y: Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 2. Quantum graph neural network utilities
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

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
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

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
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
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
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

# --------------------------------------------------------------------------- #
# 3. Quantum convolution & fully‑connected layers
# --------------------------------------------------------------------------- #

def Conv(kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127) -> qiskit.QuantumCircuit:
    """Quantum quanvolution filter using a random circuit."""
    class QuanvCircuit:
        def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
            self.n_qubits = kernel_size ** 2
            self._circuit = qiskit.QuantumCircuit(self.n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
            self._circuit.measure_all()
            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, data):
            data = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for dat in data:
                bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(dat)}
                param_binds.append(bind)
            job = qiskit.execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
            result = job.result().get_counts(self._circuit)
            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val
            return counts / (self.shots * self.n_qubits)
    backend = qiskit.Aer.get_backend("qasm_simulator")
    return QuanvCircuit(kernel_size, backend, shots, threshold)

def FCL(n_qubits: int = 1, shots: int = 100) -> qiskit.QuantumCircuit:
    """Parameterized quantum circuit that emulates a fully‑connected layer."""
    class QuantumCircuit:
        def __init__(self, n_qubits, backend, shots):
            self._circuit = qiskit.QuantumCircuit(n_qubits)
            self.theta = qiskit.circuit.Parameter("theta")
            self._circuit.h(range(n_qubits))
            self._circuit.barrier()
            self._circuit.ry(self.theta, range(n_qubits))
            self._circuit.measure_all()
            self.backend = backend
            self.shots = shots

        def run(self, thetas):
            job = qiskit.execute(
                self._circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
            result = job.result().get_counts(self._circuit)
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probabilities = counts / self.shots
            expectation = np.sum(states * probabilities)
            return np.array([expectation])
    simulator = qiskit.Aer.get_backend("qasm_simulator")
    return QuantumCircuit(n_qubits, simulator, shots)

# --------------------------------------------------------------------------- #
# 4. Main hybrid class
# --------------------------------------------------------------------------- #

class QuantumKernelMethodGen207:
    """
    Quantum‑centric counterpart of QuantumKernelMethodGen207.
    All public methods mirror the classical build for API parity.
    """
    def __init__(
        self,
        kernel_type: str = "quantum",
        qnn_arch: Sequence[int] | None = None,
        conv_kwargs: dict | None = None,
        fcl_kwargs: dict | None = None,
    ) -> None:
        self.kernel_type = kernel_type
        self.qnn_arch = list(qnn_arch) if qnn_arch else [1, 1]
        self.conv = Conv(**(conv_kwargs or {}))
        self.fcl = FCL(**(fcl_kwargs or {}))

    def compute_kernel(self, a: Sequence[Tensor], b: Sequence[Tensor]) -> np.ndarray:
        if self.kernel_type == "quantum":
            return kernel_matrix(a, b)
        raise NotImplementedError("Only quantum kernel is implemented in the quantum build.")

    def run_qnn(self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        _, unitaries, _, _ = random_network(self.qnn_arch, samples=100)
        return feedforward(self.qnn_arch, unitaries, samples)

    def build_graph(self, states: Sequence[qt.Qobj], threshold: float) -> nx.Graph:
        return fidelity_adjacency(states, threshold)

    def run_conv(self, data) -> float:
        return self.conv.run(data)

    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        return self.fcl.run(thetas)

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "Conv",
    "FCL",
    "QuantumKernelMethodGen207",
]
