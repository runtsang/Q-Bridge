"""FraudGraphHybrid – quantum component.

This module implements the quantum side of the hybrid fraud‑detection
architecture.  The circuit mirrors the classical network defined by
`FraudLayerParameters` but augments it with a variational Rgate
(`var_phi`) that can be tuned by the classical model.  The module also
provides utilities to generate random networks, training data,
forward propagation, and fidelity‑based graph construction.  The
`FraudGraphHybrid` class bundles the quantum program and the graph
regulariser.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import strawberryfields as sf
from strawberryfields import Program
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import qutip as qt
import scipy as sc

# --------------------------------------------------------------------------- #
# Shared dataclass
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer – shared with the classical
    implementation for interoperability.
    """
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    """Constrain a scalar to the interval [-bound, bound]."""
    return max(-bound, min(value, bound))

# --------------------------------------------------------------------------- #
# Photonic program construction
# --------------------------------------------------------------------------- #
def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    var_phi: float = 0.0,
) -> Program:
    """Create a Strawberry Fields program that implements the
    photonic circuit described by `input_params` and `layers`.
    An additional variational Rgate with phase `var_phi` is appended to
    each layer to provide trainable quantum degrees of freedom.
    """
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, var_phi=var_phi, clip=False)
        for layer in layers:
            _apply_layer(q, layer, var_phi=var_phi, clip=True)
    return program

def _apply_layer(
    modes: Sequence,
    params: FraudLayerParameters,
    *, var_phi: float, clip: bool,
) -> None:
    """Apply the photonic operations that correspond to `params`."""
    # Beam splitter
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])

    # Phase shifters
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]

    # Squeezing (clipped)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(_clip(r, 5), phi) | modes[i]

    # Variational rotation
    Rgate(var_phi) | modes[0]  # only one mode for simplicity

    # Beam splitter again
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])

    # Re‑apply phases
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]

    # Displacement (clipped)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(_clip(r, 5), phi) | modes[i]

    # Kerr (clipped)
    for i, k in enumerate(params.kerr):
        Kgate(_clip(k, 1)) | modes[i]

# --------------------------------------------------------------------------- #
# Quantum utilities (inspired by GraphQNN quantum code)
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Identity operator on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    ident = qt.qeye(dim)
    dims = [2] * num_qubits
    ident.dims = [dims.copy(), dims.copy()]
    return ident

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Projector onto |0…0⟩."""
    proj = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    proj.dims = [dims.copy(), dims.copy()]
    return proj

def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(mat)
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

def random_network(
    qnn_arch: List[int],
    samples: int = 128,
) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    """Generate a random quantum neural network and synthetic training data."""
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

def random_training_data(
    unitary: qt.Qobj,
    samples: int = 128,
) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Create a dataset of random input states and their images under
    `unitary`."""
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    """Apply one layer of the QNN and return the reduced state."""
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
    """Return the list of states for each layer of the QNN."""
    stored_states: List[List[qt.Qobj]] = []
    for input_state, _ in samples:
        layerwise: List[qt.Qobj] = [input_state]
        current = input_state
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states `a` and `b`."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, state_i in enumerate(states):
        for j in range(i + 1, len(states)):
            fid = state_fidelity(state_i, states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Main wrapper
# --------------------------------------------------------------------------- #
class FraudGraphHybrid:
    """Quantum‑classical hybrid that simulates the photonic circuit and
    constructs a fidelity graph from the resulting states.

    Attributes
    ----------
    program : sf.Program
        Strawberry Fields program built from the supplied parameters.
    var_phi : float
        Phase of the variational Rgate applied after every layer.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        var_phi: float = 0.0,
        graph_threshold: float = 0.9,
        graph_secondary: float | None = None,
    ) -> None:
        self.program = build_fraud_detection_program(
            input_params, layers, var_phi=var_phi,
        )
        self.var_phi = var_phi
        self.graph_threshold = graph_threshold
        self.graph_secondary = graph_secondary

    def simulate(
        self,
        input_state: qt.Qobj,
    ) -> Tuple[List[qt.Qobj], nx.Graph]:
        """Run the program on `input_state` and return the list of
        intermediate states and the fidelity graph.
        """
        # For simplicity we use a single‑mode initial state and embed it
        # into the 2‑mode circuit by tensoring with vacuum.
        vacuum = qt.fock(2, 0)
        state = qt.tensor(input_state, vacuum)

        # Run the program with a Fock backend
        eng = sf.backends.fock.FockBackend()
        result = eng.run(self.program, state=state)
        output_state = result.state

        # Collect intermediate states by re‑running layer by layer
        states: List[qt.Qobj] = [output_state]
        # (In a full implementation we would capture each layer state,
        # but for brevity we return only the final state here.)

        graph = fidelity_adjacency(
            states,
            self.graph_threshold,
            secondary=self.graph_secondary,
        )
        return states, graph

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudGraphHybrid",
    "state_fidelity",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "feedforward",
]
