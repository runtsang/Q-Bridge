"""GraphQNNGen306 – a unified quantum‑classical neural‑network framework.

The quantum implementation mirrors the classical API but uses qutip for state
propagation, torchquantum for the quanvolution layer, and qiskit for the
auto‑encoder example.  All components expose the same public names so that
scripts can switch backends by changing a single flag.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qutip as qt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# 1.  Graph‑based utilities – quantum
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Identity matrix with explicit qubit dimensions."""
    I = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    I.dims = [dims.copy(), dims.copy()]
    return I


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Zero projector with explicit qubit dimensions."""
    P0 = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    P0.dims = [dims.copy(), dims.copy()]
    return P0


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    U, _ = np.linalg.qr(mat)  # orthonormal columns
    qobj = qt.Qobj(U)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    state = qt.Qobj(vec, dims=[[2] * num_qubits, [1] * num_qubits])
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate training pairs (state, unitary*state)."""
    data = []
    n = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(n)
        data.append((state, unitary * state))
    return data


def random_network(arch: List[int], samples: int):
    """Create a random quantum circuit network and training data."""
    target_unitary = _random_qubit_unitary(arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(arch)):
        in_q = arch[layer - 1]
        out_q = arch[layer]
        layer_ops: List[qt.Qobj] = []
        for out in range(out_q):
            op = _random_qubit_unitary(in_q + 1)
            if out_q > 1:
                op = qt.tensor(_random_qubit_unitary(in_q + 1), _tensored_id(out_q - 1))
                op = _swap_registers(op, in_q, in_q + out)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return arch, unitaries, training_data, target_unitary


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
    arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj
) -> qt.Qobj:
    in_q = arch[layer - 1]
    out_q = arch[layer]
    state = qt.tensor(input_state, _tensored_zero(out_q))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(in_q))


def feedforward(
    arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Run a forward pass through the quantum network."""
    out = []
    for sample, _ in samples:
        layerwise = [sample]
        cur = sample
        for l in range(1, len(arch)):
            cur = _layer_channel(arch, unitaries, l, cur)
            layerwise.append(cur)
        out.append(layerwise)
    return out


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared overlap of two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
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
# 2.  Quantum Quanvolution
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit layer to each 2×2 patch."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                patches.append(self.measure(qdev).view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier: quantum filter + linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        feats = self.qfilter(x)
        logits = self.linear(feats)
        return F.log_softmax(logits, dim=-1)


# --------------------------------------------------------------------------- #
# 3.  Quantum regression
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample complex states |ψ> = cosθ|0…0> + e^{iϕ} sinθ|1…1>."""
    dim = 2 ** num_wires
    omega0 = np.zeros(dim, dtype=complex)
    omega0[0] = 1.0
    omega1 = np.zeros(dim, dtype=complex)
    omega1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, dim), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset returning complex state vectors and scalar targets."""

    def __init__(self, samples: int, num_wires: int) -> None:
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QModel(tq.QuantumModule):
    """Hybrid quantum‑classical regression network."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int) -> None:
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: Tensor) -> Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        feats = self.measure(qdev)
        return self.head(feats).squeeze(-1)


# --------------------------------------------------------------------------- #
# 4.  Quantum auto‑encoder (qiskit example)
# --------------------------------------------------------------------------- #
def Autoencoder() -> tq.QuantumModule:
    """Return a qiskit‑based auto‑encoder as a tq.QuantumModule."""
    from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.primitives import StatevectorSampler as Sampler
    from qiskit_machine_learning.neural_networks import SamplerQNN

    algorithm_globals.random_seed = 42

    def ansatz(num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=5)

    def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        qc.barrier()
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    num_latent, num_trash = 3, 2
    qc = auto_encoder_circuit(num_latent, num_trash)
    sampler = Sampler()
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )
    return qnn


# --------------------------------------------------------------------------- #
# 5.  GraphQNNGen306 wrapper (quantum)
# --------------------------------------------------------------------------- #
class GraphQNNGen306:
    """Unified quantum API for a graph‑based neural network."""

    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.weights: List[List[qt.Qobj]] | None = None
        self.training_data: List[Tuple[qt.Qobj, qt.Qobj]] | None = None
        self.target: qt.Qobj | None = None

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        return random_network(arch, samples)

    def initialize_random(self, samples: int):
        self.arch, self.weights, self.training_data, self.target = self.random_network(self.arch, samples)

    def feedforward(self) -> List[List[qt.Qobj]]:
        if self.weights is None or self.training_data is None:
            raise RuntimeError("Network not initialized.")
        return feedforward(self.arch, self.weights, self.training_data)

    def fidelity_graph(self, threshold: float, *, secondary: float | None = None):
        activations = self.feedforward()
        states = [act for layer in activations for act in layer]
        return fidelity_adjacency(states, threshold, secondary=secondary)


__all__ = [
    "GraphQNNGen306",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "RegressionDataset",
    "QModel",
    "Autoencoder",
]
