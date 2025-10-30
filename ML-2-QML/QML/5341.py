"""GraphQNNGen230: unified quantum interface.

Uses Qiskit, Qutip and NetworkX.  Exposes methods analogous to the classical
version, enabling side‑by‑side experiments.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple, Callable
import networkx as nx
import qutip as qt
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler, StatevectorEstimator as Estimator
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Helper functions (Qutip)
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
    unitary = np.linalg.orth(matrix)
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

# --------------------------------------------------------------------------- #
# QNN data generation (Qutip)
# --------------------------------------------------------------------------- #

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

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int,
                   input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

# --------------------------------------------------------------------------- #
# State fidelity and adjacency (Qutip)
# --------------------------------------------------------------------------- #

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
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
# Quantum Autoencoder (Qiskit)
# --------------------------------------------------------------------------- #

def Autoencoder() -> SamplerQNN:
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    def ansatz(num_qubits):
        return RealAmplitudes(num_qubits, reps=5)

    def auto_encoder_circuit(num_latent, num_trash):
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()
        auxiliary_qubit = num_latent + 2 * num_trash
        circuit.h(auxiliary_qubit)
        for i in range(num_trash):
            circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
        circuit.h(auxiliary_qubit)
        circuit.measure(auxiliary_qubit, cr[0])
        return circuit

    num_latent = 3
    num_trash = 2
    circuit = auto_encoder_circuit(num_latent, num_trash)

    def identity_interpret(x):
        return x

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

# --------------------------------------------------------------------------- #
# Fast estimator (Qiskit)
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
# QCNN (Qiskit)
# --------------------------------------------------------------------------- #

def QCNN() -> EstimatorQNN:
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    def conv_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    circuit = conv_layer(4, "θ")

    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc

    sources = [0, 1]
    sinks = [2, 3]
    circuit = pool_layer(sources, sinks, "θ")

    feature_map = ZFeatureMap(8)

    ansatz = QuantumCircuit(8, name="Ansatz")

    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

# --------------------------------------------------------------------------- #
# Unified GraphQNNGen230 class (quantum)
# --------------------------------------------------------------------------- #

class GraphQNNGen230:
    """
    Quantum counterpart of the classical GraphQNNGen230.  All heavy lifting is
    performed by Qiskit / Qutip primitives; the public API mirrors the
    classical version for seamless side‑by‑side comparison.
    """

    def __init__(self) -> None:
        self.qnn_arch: List[int] | None = None
        self.unitaries: List[List[qt.Qobj]] | None = None
        self.training_data: List[Tuple[qt.Qobj, qt.Qobj]] | None = None
        self.target_unitary: qt.Qobj | None = None
        self.autoencoder_qnn: SamplerQNN | None = None
        self.estimator_obj: FastBaseEstimator | None = None
        self.qcnn_qnn: EstimatorQNN | None = None

    # ------------------------------------------------------------------ #
    # QNN helpers
    # ------------------------------------------------------------------ #

    def build_qnn(self, qnn_arch: Sequence[int], samples: int) -> None:
        self.qnn_arch, self.unitaries, self.training_data, self.target_unitary = random_network(qnn_arch, samples)

    def feedforward(self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        if self.unitaries is None:
            raise RuntimeError("QNN not built – call build_qnn first.")
        return feedforward(self.qnn_arch, self.unitaries, samples)

    def fidelity_adjacency(self, threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        if self.unitaries is None:
            raise RuntimeError("QNN not built – call build_qnn first.")
        activations = self.feedforward(self.training_data)
        states = [act[-1] for act in activations]
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                  secondary_weight=secondary_weight)

    # ------------------------------------------------------------------ #
    # Autoencoder helpers
    # ------------------------------------------------------------------ #

    def build_autoencoder(self) -> None:
        self.autoencoder_qnn = Autoencoder()

    def get_autoencoder(self) -> SamplerQNN:
        if self.autoencoder_qnn is None:
            raise RuntimeError("Autoencoder not built – call build_autoencoder first.")
        return self.autoencoder_qnn

    # ------------------------------------------------------------------ #
    # Estimator helpers
    # ------------------------------------------------------------------ #

    def build_estimator(self, circuit: QuantumCircuit) -> None:
        self.estimator_obj = FastBaseEstimator(circuit)

    def evaluate_estimator(self, observables: Iterable[BaseOperator],
                           parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        if self.estimator_obj is None:
            raise RuntimeError("Estimator not built – call build_estimator first.")
        return self.estimator_obj.evaluate(observables, parameter_sets)

    # ------------------------------------------------------------------ #
    # QCNN helpers
    # ------------------------------------------------------------------ #

    def build_qcnn(self) -> None:
        self.qcnn_qnn = QCNN()

    def qcnn_forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.qcnn_qnn is None:
            raise RuntimeError("QCNN not built – call build_qcnn first.")
        # EstimatorQNN expects a 1‑D array of feature values
        return np.array(self.qcnn_qnn.forward(inputs))

    __all__ = [
        "GraphQNNGen230",
        "Autoencoder",
        "FastBaseEstimator",
        "QCNN",
    ]
