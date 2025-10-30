import torch
import torchquantum as tq
import numpy as np
import networkx as nx
import itertools
from collections.abc import Iterable, Sequence
from typing import Tuple, List, Iterable, Sequence
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# ----- Quantum regression utilities -----
def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0…0> + e^{iφ} sin(theta)|1…1>."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset for quantum superposition regression."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ----- Fast estimator for quantum circuits -----
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# ----- Core quantum graph neural network -----
def _tensored_id(num_qubits: int):
    import qutip as qt
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int):
    import qutip as qt
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector

def _swap_registers(op, source, target):
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int):
    import qutip as qt
    import scipy as sc
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int):
    import qutip as qt
    import scipy as sc
    dim = 2 ** num_qubits
    amps = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amps /= sc.linalg.norm(amps)
    state = qt.Qobj(amps)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: "qt.Qobj", samples: int):
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: list[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    unitaries: list[list["qt.Qobj"]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list["qt.Qobj"] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)
    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state, keep):
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state, remove):
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch, unitaries, layer, input_state):
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch, unitaries, samples):
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a, b):
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states, threshold, *, secondary=None, secondary_weight=0.5):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ----- Quantum Graph Neural Network class -----
class QLayer(tq.QuantumModule):
    """Variational layer that applies a random unitary followed by single‑qubit rotations."""
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_inputs + num_outputs)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for wire in range(self.num_inputs + self.num_outputs):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

class GraphQNN(tq.QuantumModule):
    """Quantum graph neural network mirroring the classical GraphQNN API."""
    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = tuple(arch)
        self.layers: list[tq.QuantumModule] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            self.layers.append(QLayer(in_f, out_f))

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.arch[-1], bsz=bsz, device=state_batch.device)
        encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{self.arch[-1]}xRy"])
        encoder(qdev, state_batch)
        for layer in self.layers:
            layer(qdev)
        measure = tq.MeasureAll(tq.PauliZ)
        features = measure(qdev)
        return features.squeeze(-1)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        base_circuit = self.to_circuit()
        estimator = FastBaseEstimator(base_circuit)
        return estimator.evaluate(observables, parameter_sets)

__all__ = [
    "GraphQNN",
    "RegressionDataset",
    "generate_superposition_data",
    "FastBaseEstimator",
    "state_fidelity",
    "fidelity_adjacency",
]
