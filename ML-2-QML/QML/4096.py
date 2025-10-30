from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List, Dict, Any

import numpy as np
import networkx as nx
import qutip as qt

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from dataclasses import dataclass

# ----------------------------------------
# Photonic fraud‑detection parameters
# ----------------------------------------
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

# ----------------------------------------
# Graph‑based quantum neural network utilities
# ----------------------------------------
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

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
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
    samples: Iterable[tuple[qt.Qobj, qt.Qobj]],
):
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
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
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

# ----------------------------------------
# Quantum autoencoder
# ----------------------------------------
def QuantumAutoencoder(num_latent: int, num_trash: int) -> SamplerQNN:
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

    qc = auto_encoder_circuit(num_latent, num_trash)
    def identity_interpret(x):
        return x
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

# ----------------------------------------
# Unified hybrid quantum model
# ----------------------------------------
class FraudDetectionHybridModel:
    """Unified quantum hybrid model combining photonic fraud detection, graph‑based QNN, and variational autoencoder."""
    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        graph_arch: Sequence[int],
        autoencoder_params: Dict[str, Any],
    ) -> None:
        self.fraud_program = build_fraud_detection_program(fraud_params, fraud_layers)
        self.graph_arch = list(graph_arch)
        self.autoencoder = QuantumAutoencoder(**autoencoder_params)

    def run_fraud_detection(
        self,
        inputs: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Simulate the StrawberryFields photonic circuit for given inputs."""
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})
        results = []
        for inp in inputs:
            prog = self.fraud_program
            result = eng.run(prog)
            results.append(result.state)
        return np.array([r.data for r in results])

    def run_autoencoder(
        self,
        inputs: List[float],
    ) -> np.ndarray:
        """Evaluate the variational autoencoder on input data."""
        qnn = self.autoencoder
        input_vector = np.zeros(len(qnn.input_params))
        output = qnn.sample(input_vector)
        return output

    def compute_graph_fidelity(
        self,
        samples: Iterable[tuple[qt.Qobj, qt.Qobj]],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Create a fidelity‑based adjacency graph from quantum states."""
        states = [s for s, _ in samples]
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "QuantumAutoencoder",
    "FraudDetectionHybridModel",
]
