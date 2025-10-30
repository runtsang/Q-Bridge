"""Hybrid Graph Neural Network for quantum circuits.

This module extends the original GraphQNN utilities by adding a
HybridGraphQNN class that can mix classical feed‑forward layers,
quantum sampler circuits, photonic fraud detection programs, and
QCNN ansatz circuits.  The interface mirrors the classical
implementation but all layers are defined with Qiskit or Strawberry
Fields, and the forward pass propagates a quantum state through each
layer.
"""

import itertools
import random
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Union

import networkx as nx
import qutip as qt
import scipy as sc
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import Operator
from qiskit_machine_learning.utils import algorithm_globals
from dataclasses import dataclass

# --- Helper functions ------------------------------------------------------------

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
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --- Quantum layer implementations -----------------------------------------------

class SamplerQNN:
    """Parameterised quantum sampler used as a layer."""
    def __init__(self, num_qubits: int = 2) -> None:
        self.num_qubits = num_qubits
        self.inputs = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("w", num_qubits * 2)
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[0], 0)
        self.circuit.ry(self.weights[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[2], 0)
        self.circuit.ry(self.weights[3], 1)

    def forward(self, state: qt.Qobj) -> qt.Qobj:
        unitary = Operator(self.circuit).data
        new_state = qt.Qobj(unitary @ state.data, dims=state.dims)
        return new_state

@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic fraud detection layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

class FraudDetectionProgram:
    """Photonic fraud detection program using Strawberry Fields."""
    def __init__(self, params: FraudLayerParameters) -> None:
        self.params = params
        self.program = sf.Program(2)
        with self.program.context as q:
            _apply_layer(q, params, clip=False)

    def forward(self, state: qt.Qobj) -> qt.Qobj:
        unitary = qt.Qobj(self.program.get_unitary())
        unitary.dims = [state.dims[0], state.dims[0]]
        return unitary * state

class QCNNQuantum:
    """Quantum convolutional neural network ansatz using Qiskit."""
    def __init__(self, num_qubits: int = 8) -> None:
        self.num_qubits = num_qubits
        self.feature_map = ZFeatureMap(num_qubits)
        self.ansatz = self._build_ansatz(num_qubits)

    def _build_ansatz(self, num_qubits: int) -> QuantumCircuit:
        ansatz = QuantumCircuit(num_qubits)
        for i in range(0, num_qubits, 2):
            ansatz.cx(i, i+1)
            ansatz.ry(0.1, i)
            ansatz.ry(0.2, i+1)
        for i in range(0, num_qubits//2, 2):
            ansatz.cx(i, i+1)
        return ansatz

    def forward(self, state: qt.Qobj) -> qt.Qobj:
        unitary = Operator(self.ansatz).data
        new_state = qt.Qobj(unitary @ state.data, dims=state.dims)
        return new_state

# --- Hybrid network ---------------------------------------------------------------

class HybridGraphQNN:
    """Hybrid quantum graph neural network that stitches together quantum
    and classical‑inspired layers.  The interface mirrors the classical
    implementation but all layers are defined with Qiskit or Strawberry
    Fields, and the forward pass propagates a quantum state.
    """
    def __init__(self,
                 architecture: Sequence[int],
                 layer_types: Sequence[str],
                 seed: int | None = None) -> None:
        if seed is not None:
            random.seed(seed)
            algorithm_globals.random_seed = seed
        self.architecture = list(architecture)
        self.layer_types = list(layer_types)
        if len(self.layer_types)!= len(self.architecture) - 1:
            raise ValueError("layer_types length must be architecture length minus one")
        self.layers: List[Union[qt.Qobj, SamplerQNN, FraudDetectionProgram, QCNNQuantum]] = []
        self._build_layers()

    def _build_layers(self) -> None:
        for idx, ltype in enumerate(self.layer_types):
            in_f, out_f = self.architecture[idx], self.architecture[idx+1]
            if ltype == "feedforward":
                unitary = _random_qubit_unitary(in_f)
                self.layers.append(unitary)
            elif ltype == "sampler":
                self.layers.append(SamplerQNN(num_qubits=in_f))
            elif ltype == "fraud":
                params = _random_fraud_params()
                self.layers.append(FraudDetectionProgram(params))
            elif ltype == "qcnn":
                self.layers.append(QCNNQuantum(num_qubits=out_f))
            else:
                raise ValueError(f"Unsupported layer type: {ltype}")

    def forward(self, state: qt.Qobj) -> qt.Qobj:
        current = state
        for layer in self.layers:
            if isinstance(layer, qt.Qobj):
                current = layer * current
            else:
                current = layer.forward(current)
        return current

    def random_network(self, samples: int) -> Tuple[List[int], List[Union[qt.Qobj, SamplerQNN, FraudDetectionProgram, QCNNQuantum]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        unitaries: List[Union[qt.Qobj, SamplerQNN, FraudDetectionProgram, QCNNQuantum]] = []
        for in_f, out_f in zip(self.architecture[:-1], self.architecture[1:]):
            unitaries.append(_random_qubit_unitary(in_f))
        target_unitary = _random_qubit_unitary(self.architecture[-1])
        training_data = random_training_data(target_unitary, samples)
        return list(self.architecture), unitaries, training_data, target_unitary

    def feedforward(self,
                    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        stored: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            activations = [sample]
            current = sample
            for layer in self.layers:
                if isinstance(layer, qt.Qobj):
                    current = layer * current
                else:
                    current = layer.forward(current)
                activations.append(current)
            stored.append(activations)
        return stored

    def state_fidelity(self, a: qt.Qobj, b: qt.Qobj) -> float:
        return state_fidelity(a, b)

    def fidelity_adjacency(self,
                           states: Sequence[qt.Qobj],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                   secondary_weight=secondary_weight)

__all__ = [
    "HybridGraphQNN",
    "SamplerQNN",
    "FraudDetectionProgram",
    "QCNNQuantum",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
