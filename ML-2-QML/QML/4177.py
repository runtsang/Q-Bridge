"""Quantum implementation of GraphQNNGen157.

The quantum module mirrors the classical interface but operates on
quantum states and unitaries.  It supports:
- random quantum networks
- state‑fidelity based graphs
- quantum kernel evaluation via a fixed TorchQuantum ansatz
- fraud‑detection Strawberry Fields program generation
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import networkx as nx
import numpy as np
import qutip as qt
import scipy as sc
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

Tensor = torch.Tensor
QObj = qt.Qobj


def _tensored_identity(num_qubits: int) -> QObj:
    """Return the identity operator on ``num_qubits`` qubits."""
    I = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    I.dims = [dims.copy(), dims.copy()]
    return I


def _tensored_zero(num_qubits: int) -> QObj:
    """Return the zero‑state projector on ``num_qubits`` qubits."""
    zero = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    zero.dims = [dims.copy(), dims.copy()]
    return zero


def _swap_registers(op: QObj, src: int, tgt: int) -> QObj:
    if src == tgt:
        return op
    order = list(range(len(op.dims[0])))
    order[src], order[tgt] = order[tgt], order[src]
    return op.permute(order)


def _rand_unitary(num_qubits: int) -> QObj:
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    u = sc.linalg.orth(mat)
    qobj = qt.Qobj(u)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _rand_state(num_qubits: int) -> QObj:
    dim = 2 ** num_qubits
    amp = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amp /= sc.linalg.norm(amp)
    state = qt.Qobj(amp)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data_q(unitary: QObj, samples: int) -> List[Tuple[QObj, QObj]]:
    """Generate training pairs by applying ``unitary`` to random states."""
    data: List[Tuple[QObj, QObj]] = []
    nq = len(unitary.dims[0])
    for _ in range(samples):
        st = _rand_state(nq)
        data.append((st, unitary * st))
    return data


def random_network_q(arch: List[int], samples: int) -> Tuple[List[int], List[List[QObj]], List[Tuple[QObj, QObj]], QObj]:
    """Create a random quantum network and its training data."""
    target = _rand_unitary(arch[-1])
    train = random_training_data_q(target, samples)

    ops: List[List[QObj]] = [[]]
    for layer in range(1, len(arch)):
        nin, nout = arch[layer - 1], arch[layer]
        layer_ops: List[QObj] = []
        for out_idx in range(nout):
            op = _rand_unitary(nin + 1)
            if nout > 1:
                op = qt.tensor(_rand_unitary(nin + 1), _tensored_identity(nout - 1))
                op = _swap_registers(op, nin, nin + out_idx)
            layer_ops.append(op)
        ops.append(layer_ops)

    return arch, ops, train, target


def _partial_trace_keep(state: QObj, keep: Sequence[int]) -> QObj:
    if len(keep) == len(state.dims[0]):
        return state
    return state.ptrace(list(keep))


def _partial_trace_remove(state: QObj, remove: Sequence[int]) -> QObj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    arch: Sequence[int], ops: Sequence[Sequence[QObj]], layer: int, inp_state: QObj
) -> QObj:
    nin, nout = arch[layer - 1], arch[layer]
    state = qt.tensor(inp_state, _tensored_zero(nout))

    unitary = ops[layer][0].copy()
    for g in ops[layer][1:]:
        unitary = g * unitary

    return _partial_trace_remove(unitary * state * unitary.dag(), range(nin))


def feedforward_q(
    arch: Sequence[int], ops: Sequence[Sequence[QObj]], samples: Iterable[Tuple[QObj, QObj]]
) -> List[List[QObj]]:
    """Run a quantum forward pass and collect intermediate states."""
    outputs: List[List[QObj]] = []
    for st, _ in samples:
        layerwise: List[QObj] = [st]
        cur = st
        for layer in range(1, len(arch)):
            cur = _layer_channel(arch, ops, layer, cur)
            layerwise.append(cur)
        outputs.append(layerwise)
    return outputs


def _state_fidelity_q(a: QObj, b: QObj) -> float:
    """Squared overlap of two pure quantum states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_graph_q(
    states: Sequence[QObj], thr: float, *, sec: Optional[float] = None, sec_w: float = 0.5
) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity_q(s_i, s_j)
        if fid >= thr:
            g.add_edge(i, j, weight=1.0)
        elif sec is not None and fid >= sec:
            g.add_edge(i, j, weight=sec_w)
    return g


@dataclass
class FraudLayerParams:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def build_fraud_detection_program_q(
    inp_params: FraudLayerParams, layers: Iterable[FraudLayerParams]
) -> "sf.Program":
    """Create a Strawberry Fields program that mirrors the photonic circuit."""
    try:
        import strawberryfields as sf
        from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
    except Exception as exc:
        raise RuntimeError("StrawberryFields is not available.") from exc

    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, inp_params, clip=False)
        for l in layers:
            _apply_layer(q, l, clip=True)
    return prog


def _apply_layer(modes: Sequence, params: FraudLayerParams, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, ph in enumerate(params.phases):
        Rgate(ph) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, ph in enumerate(params.phases):
        Rgate(ph) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class KernalAnsatz(tq.QuantumModule):
    """Quantum RBF kernel ansatz implemented with TorchQuantum."""

    def __init__(self, func_list: List[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: Tensor, y: Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class KernelQML(tq.QuantumModule):
    """Fixed quantum kernel that returns the absolute overlap of the final state."""

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


def kernel_matrix_q(a: Sequence[QObj], b: Sequence[QObj]) -> np.ndarray:
    """Compute a quantum kernel Gram matrix between two state sets."""
    return np.array([[abs((a_i.dag() * b_j)[0, 0]) ** 2 for b_j in b] for a_i in a])


class GraphQNNGen157:
    """Quantum counterpart of :class:`~ml.GraphQNNGen157`."""

    def __init__(self, arch: Sequence[int], seed: Optional[int] = None) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            sc.random.seed(seed)
        self.arch = tuple(arch)
        self.ops, self.train_data, self.target = random_network_q(self.arch, nsamples=100)

    # --------------------------------------------------------------------- #
    # Quantum utilities
    # --------------------------------------------------------------------- #
    def forward_q(self, inputs: QObj) -> List[List[QObj]]:
        """Return intermediate quantum states for a batch of input states."""
        return feedforward_q(self.arch, self.ops, [(inputs, self.target * inputs)])

    def fidelity_graph_q(
        self,
        states: Sequence[QObj],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Graph of state similarities in the quantum domain."""
        return fidelity_graph_q(states, threshold, sec=secondary, sec_w=secondary_weight)

    def kernel(self, a: Sequence[QObj], b: Sequence[QObj]) -> np.ndarray:
        """Compute the quantum kernel matrix."""
        return kernel_matrix_q(a, b)

    # --------------------------------------------------------------------- #
    # Fraud‑detection utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def fraud_program(
        inp_params: FraudLayerParams,
        layers: Iterable[FraudLayerParams],
    ) -> "sf.Program":
        return build_fraud_detection_program_q(inp_params, layers)

    # --------------------------------------------------------------------- #
    # Classical placeholders for API compatibility
    # --------------------------------------------------------------------- #
    def classical_forward(self, *args, **kwargs) -> None:
        """Placeholder for classical forward – to be overridden in ML module."""
        raise NotImplementedError("Classical forward not implemented in quantum module.")


__all__ = [
    "GraphQNNGen157",
    "FraudLayerParams",
    "build_fraud_detection_program_q",
]
