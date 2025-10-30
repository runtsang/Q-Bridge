import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import qiskit.circuit.library as lib
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.providers.aer import AerSimulator
import qiskit.utils as qiskit_utils

import networkx as nx
import numpy as np
import qutip as qt
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List

# ----------------------------------------------------------------------
# Photonic layer parameters (same as the classical counterpart)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 2‑mode photonic program
# ----------------------------------------------------------------------
def _apply_layer(q, params: FraudLayerParameters, *, clip: bool) -> None:
    # Beam‑splitter
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    # Phase rotations
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    # Squeezing
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | q[i]
    # Second beam‑splitter
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    # Second phase rotations
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    # Displacements
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | q[i]
    # Kerr gates
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | q[i]


def build_photonic_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Return a Strawberry Fields program that mirrors the photonic fraud‑detection circuit."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


# ----------------------------------------------------------------------
# Quantum auto‑encoder (swap‑test)
# ----------------------------------------------------------------------
def quantum_autoencoder(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Return a Qiskit circuit that implements a swap‑test based quantum auto‑encoder."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Ansatz
    qc.compose(lib.RealAmplitudes(num_latent + num_trash, reps=5), range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Swap‑test auxiliary qubit
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc


# ----------------------------------------------------------------------
# Helper functions for the graph‑QNN (reference[3] QML)
# ----------------------------------------------------------------------
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


def random_network(qnn_arch: Sequence[int], samples: int):
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
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities.

    Edges with fidelity >= ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and ``threshold`` are added with ``secondary_weight``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# ----------------------------------------------------------------------
# High‑level wrapper that ties the components together
# ----------------------------------------------------------------------
class FraudNetQML:
    """Hybrid quantum front‑end that combines the photonic circuit, a swap‑test auto‑encoder,
    and a fidelity‑based graph.  The class is intentionally lightweight – it merely
    constructs the circuits and provides convenience methods for execution."""
    def __init__(
        self,
        photonic_params: FraudLayerParameters,
        photonic_layers: Iterable[FraudLayerParameters],
        num_latent: int,
        num_trash: int,
    ) -> None:
        self.photonic_program = build_photonic_program(photonic_params, photonic_layers)
        self.autoencoder_circuit = quantum_autoencoder(num_latent, num_trash)

    def run_photonic(self, engine: sf.Engine | None = None) -> qt.Qobj:
        """Execute the photonic program on a Strawberry Fields engine."""
        if engine is None:
            engine = sf.Engine("fock", backend_options={"cutoff_dim": 5})
        results = engine.run(self.photonic_program)
        return results.state

    def run_autoencoder(self, backend: qiskit.providers.Backend | None = None) -> Statevector:
        """Execute the quantum auto‑encoder on a Qiskit backend."""
        if backend is None:
            backend = AerSimulator()
        job = backend.run(self.autoencoder_circuit)
        result = job.result()
        return Statevector.from_dict(result.get_counts(self.autoencoder_circuit))

    def graph_from_states(self, states: Sequence[qt.Qobj], threshold: float) -> nx.Graph:
        """Build a graph from fidelities of a list of qutip states."""
        return fidelity_adjacency(states, threshold)


__all__ = [
    "FraudLayerParameters",
    "build_photonic_program",
    "quantum_autoencoder",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "FraudNetQML",
]
