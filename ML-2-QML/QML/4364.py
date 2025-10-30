"""Hybrid quantum GraphQNN implementation integrating Qiskit attention, EstimatorQNN, and Strawberry Fields photonic layers."""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import networkx as nx
import qiskit
import qutip as qt
import strawberryfields as sf
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from strawberryfields import ops

# --------------------------------------------------------------------------- #
# Basic quantum utilities (from original GraphQNN QML seed)
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
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.qr(matrix)[0]
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes /= np.linalg.norm(amplitudes)
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
# Quantum Self‑Attention (Qiskit)
# --------------------------------------------------------------------------- #
def _build_attention_circuit(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(np.random.rand() * 2 * np.pi, i)
        qc.ry(np.random.rand() * 2 * np.pi, i)
        qc.rz(np.random.rand() * 2 * np.pi, i)
    for i in range(num_qubits - 1):
        qc.crx(np.random.rand() * 2 * np.pi, i, i + 1)
    return qc


# --------------------------------------------------------------------------- #
# Fraud‑Detection photonic program (Strawberry Fields)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    ops.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        ops.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        ops.Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    ops.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        ops.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        ops.Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        ops.Kgate(k if not clip else _clip(k, 1)) | modes[i]


# --------------------------------------------------------------------------- #
# Quantum GraphQNN class
# --------------------------------------------------------------------------- #
class GraphQNN:
    """
    Quantum version of GraphQNN that chains random unitaries per layer,
    optionally inserts a Qiskit self‑attention circuit, evaluates an
    EstimatorQNN expectation, and can append a Strawberry Fields photonic
    program for each layer.
    """
    def __init__(
        self,
        arch: Sequence[int],
        use_attention: bool = True,
        use_estimator: bool = True,
        use_photonic: bool = False,
    ) -> None:
        self.arch = list(arch)
        self.use_attention = use_attention
        self.use_estimator = use_estimator
        self.use_photonic = use_photonic

        self.unitaries: List[List[qt.Qobj]] = []
        self.attention_circuits: List[qt.Qobj] = []
        self.estimators: List[EstimatorQNN] = []
        self.photonic_programs: List[sf.Program] = []

        for layer in range(1, len(arch)):
            num_inputs = arch[layer - 1]
            num_outputs = arch[layer]
            layer_ops: List[qt.Qobj] = []
            for output in range(num_outputs):
                op = _random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                    op = _swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            self.unitaries.append(layer_ops)

            if self.use_attention:
                qc = _build_attention_circuit(num_inputs + 1)
                att_qobj = qt.Qobj(qc.to_matrix(), dims=[ [2] * (num_inputs + 1), [2] * (num_inputs + 1) ])
                self.attention_circuits.append(att_qobj)

            if self.use_estimator:
                estimator = EstimatorQNN()
                self.estimators.append(estimator)

            if self.use_photonic:
                program = build_fraud_detection_program(
                    FraudLayerParameters(
                        bs_theta=random.random(),
                        bs_phi=random.random(),
                        phases=(random.random(), random.random()),
                        squeeze_r=(random.random(), random.random()),
                        squeeze_phi=(random.random(), random.random()),
                        displacement_r=(random.random(), random.random()),
                        displacement_phi=(random.random(), random.random()),
                        kerr=(random.random(), random.random()),
                    ),
                    [],
                )
                self.photonic_programs.append(program)

        self.target_unitary = _random_qubit_unitary(arch[-1])

    def random_network(self, samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        """Re‑generate random unitaries, attention circuits, and training data."""
        return random_network(self.arch, samples)

    def random_training_data(self, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        return random_training_data(self.target_unitary, samples)

    def feedforward(self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        stored_states: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(self.arch)):
                current_state = _layer_channel(self.arch, self.unitaries, layer, current_state)
                if self.use_attention:
                    current_state = self.attention_circuits[layer - 1] * current_state
                if self.use_photonic:
                    engine = sf.Engine("gaussian")
                    result = engine.run(self.photonic_programs[layer - 1], [current_state])
                    current_state = result.state
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    def state_fidelity(self, a: qt.Qobj, b: qt.Qobj) -> float:
        return state_fidelity(a, b)

    def fidelity_adjacency(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "_tensored_id",
    "_tensored_zero",
    "_swap_registers",
    "_random_qubit_unitary",
    "_random_qubit_state",
    "random_training_data",
    "random_network",
    "_partial_trace_keep",
    "_partial_trace_remove",
    "_layer_channel",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "_apply_layer",
    "GraphQNN",
]
