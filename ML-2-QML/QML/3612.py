"""GraphQNNFusion – quantum implementation.

Mirrors the classical interface while using quantum state propagation
and a Strawberry‑Fields photonic fraud circuit.  The class name and
method signatures are identical to the classical counterpart for
drop‑in compatibility."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import qutip as qt
import scipy as sc
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

Qobj = qt.Qobj

# --------------------------------------------------------------------------- #
# 1. Fraud‑style parameter container (identical to classical)
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

# --------------------------------------------------------------------------- #
# 2. Helper functions for quantum state handling
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> Qobj:
    id_mat = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    id_mat.dims = [dims.copy(), dims.copy()]
    return id_mat

def _tensored_zero(num_qubits: int) -> Qobj:
    zero = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    zero.dims = [dims.copy(), dims.copy()]
    return zero

def _swap_registers(op: Qobj, source: int, target: int) -> Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> Qobj:
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(mat)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> Qobj:
    dim = 2 ** num_qubits
    amp = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amp /= sc.linalg.norm(amp)
    state = qt.Qobj(amp)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

# --------------------------------------------------------------------------- #
# 3. Training data generation for a unitary
# --------------------------------------------------------------------------- #
def random_training_data(unitary: Qobj, samples: int) -> List[Tuple[Qobj, Qobj]]:
    """Generate (input, target) state pairs for a target unitary."""
    data = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        data.append((state, unitary * state))
    return data

# --------------------------------------------------------------------------- #
# 4. Random quantum network (unitary chain)
# --------------------------------------------------------------------------- #
def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a chain of random unitaries and a training set for the last layer."""
    target = _random_qubit_unitary(qnn_arch[-1])
    training = random_training_data(target, samples)
    unitaries: List[List[Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_sz = qnn_arch[layer - 1]
        out_sz = qnn_arch[layer]
        layer_ops: List[Qobj] = []
        for _ in range(out_sz):
            op = _random_qubit_unitary(in_sz + 1)
            if out_sz > 1:
                op = qt.tensor(_random_qubit_unitary(in_sz + 1), _tensored_id(out_sz - 1))
                op = _swap_registers(op, in_sz, in_sz + _)
            layer_ops.append(op)
        unitaries.append(layer_ops)
    return list(qnn_arch), unitaries, training, target

# --------------------------------------------------------------------------- #
# 5. Layer‑wise channel application
# --------------------------------------------------------------------------- #
def _partial_trace_keep(state: Qobj, keep: Sequence[int]) -> Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: Qobj, remove: Sequence[int]) -> Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)

def _layer_channel(arch: Sequence[int],
                   unitaries: Sequence[Sequence[Qobj]],
                   layer: int,
                   input_state: Qobj) -> Qobj:
    in_sz = arch[layer - 1]
    out_sz = arch[layer]
    state = qt.tensor(input_state, _tensored_zero(out_sz))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    out_state = layer_unitary * state * layer_unitary.dag()
    return _partial_trace_remove(out_state, range(in_sz))

# --------------------------------------------------------------------------- #
# 6. Feed‑forward propagation
# --------------------------------------------------------------------------- #
def feedforward(arch: Sequence[int],
                unitaries: Sequence[Sequence[Qobj]],
                samples: Iterable[Tuple[Qobj, Qobj]]) -> List[List[Qobj]]:
    """Return the state at each layer for every sample."""
    all_states = []
    for sample, _ in samples:
        layer_states = [sample]
        current = sample
        for layer in range(1, len(arch)):
            current = _layer_channel(arch, unitaries, layer, current)
            layer_states.append(current)
        all_states.append(layer_states)
    return all_states

# --------------------------------------------------------------------------- #
# 7. Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: Qobj, b: Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[Qobj],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
# 8. Photonic fraud circuit construction
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Return a Strawberry Fields program that reproduces the classical fraud circuit."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

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

# --------------------------------------------------------------------------- #
# 9. Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "FraudLayerParameters",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "build_fraud_detection_program",
]
