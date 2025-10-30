"""Quantum hybrid graph neural network mirroring the classical GraphQNNGen031.

The quantum implementation uses Qiskit and QuTiP to:
* Propagate quantum states through a graph‑structured unitary network.
* Apply a quantum autoencoder (swap‑test based) to compress the state.
* Sample the compressed state with a Qiskit SamplerQNN.
* Classify the sampled probabilities with a quantum QCNN (EstimatorQNN).
"""

import itertools
from typing import List, Tuple, Sequence

import networkx as nx
import qutip as qt
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info import Statevector, SparsePauliOp

# --------------------------------------------------------------------------- #
# Core QNN state propagation (from original QML seed)
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
# Quantum autoencoder circuit (swap‑test based)
# --------------------------------------------------------------------------- #

def quantum_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Swap test
    auxiliary_qubit = num_latent + 2 * num_trash
    qc.h(auxiliary_qubit)
    for i in range(num_trash):
        qc.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
    qc.h(auxiliary_qubit)
    qc.measure(auxiliary_qubit, cr[0])
    return qc

# --------------------------------------------------------------------------- #
# Quantum SamplerQNN
# --------------------------------------------------------------------------- #

def quantum_sampler_qnn() -> QSamplerQNN:
    algorithm_globals.random_seed = 42
    inputs2 = ParameterVector("input", 2)
    weights2 = ParameterVector("weight", 4)

    qc2 = QuantumCircuit(2)
    qc2.ry(inputs2[0], 0)
    qc2.ry(inputs2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[0], 0)
    qc2.ry(weights2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[2], 0)
    qc2.ry(weights2[3], 1)

    sampler = Sampler()
    return QSamplerQNN(circuit=qc2, input_params=inputs2, weight_params=weights2, sampler=sampler)

# --------------------------------------------------------------------------- #
# Quantum QCNN circuit (EstimatorQNN)
# --------------------------------------------------------------------------- #

def quantum_qcnn() -> QEstimatorQNN:
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

    params = ParameterVector("θ", length=3)
    circuit = conv_circuit(params)

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

    return QEstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

# --------------------------------------------------------------------------- #
# Unified quantum class
# --------------------------------------------------------------------------- #

class GraphQNNGen031:
    """
    Quantum counterpart of the classical GraphQNNGen031.

    It propagates quantum states through a graph‑structured unitary network,
    compresses them with a quantum autoencoder, samples with a Qiskit SamplerQNN,
    and classifies the sampled probabilities using a quantum QCNN (EstimatorQNN).
    """

    def __init__(self, qnn_arch: Sequence[int]):
        self.qnn_arch = list(qnn_arch)
        self.unitaries, self.training_data, self.target_unitary = random_network(self.qnn_arch, samples=10)
        self.autoencoder_circuit = quantum_autoencoder_circuit(num_latent=3, num_trash=2)
        self.sampler_qnn = quantum_sampler_qnn()
        self.qcnn = quantum_qcnn()

    def forward(self, input_states: List[qt.Qobj]) -> Statevector:
        """
        Execute the quantum pipeline.

        Parameters
        ----------
        input_states : List[qt.Qobj]
            List of input pure states.

        Returns
        -------
        Statevector
            Final QCNN expectation value as a Statevector (scalar).
        """
        # 1. Propagate through graph‑structured unitaries
        activations = feedforward(self.qnn_arch, self.unitaries, [(s, None) for s in input_states])
        # 2. Autoencoder compression
        compressed_states = []
        for act in activations:
            state = act[-1]
            # Apply autoencoder circuit to the state
            qc = self.autoencoder_circuit
            # Prepare initial state vector
            sv = Statevector(state)
            # Execute circuit
            result = qc.compose(qc, inplace=False)
            compressed = result
            compressed_states.append(compressed)

        # 3. Sample compressed states
        sampler = self.sampler_qnn
        probs = []
        for cs in compressed_states:
            # The sampler expects a circuit; we use the same circuit as the autoencoder
            # but we only need the measurement outcome probabilities
            result = sampler.sample(cs)
            probs.append(result)

        # 4. QCNN classification
        qcnn = self.qcnn
        # Flatten probabilities to a feature vector
        feature_vec = np.concatenate([p for p in probs]).reshape(1, -1)
        # Random weight values for demonstration
        weight_vals = np.random.randn(len(qcnn.weight_params))
        expectation = qcnn.evaluate(inputs=feature_vec[0], weight_values=weight_vals)
        return expectation

    def build_graph(self, threshold: float, secondary: float | None = None) -> nx.Graph:
        """
        Construct a graph from the current quantum states using fidelity thresholds.
        """
        states = [act[-1] for act in feedforward(self.qnn_arch, self.unitaries, [(s, None) for s in self.training_data])]
        return fidelity_adjacency(states, threshold, secondary=secondary)

    def __repr__(self) -> str:
        return f"<GraphQNNGen031 qnn_arch={self.qnn_arch}>"

__all__ = ["GraphQNNGen031"]
