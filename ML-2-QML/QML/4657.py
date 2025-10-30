"""Quantum‑centric Quanvolution network with graph utilities.

The quantum module mirrors the classical structure but replaces the
patch‑wise feature extraction with a variational quantum kernel.
It also exposes a quantum ansatz for classification and fidelity‑based
graph construction on quantum states."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import qutip as qt
import torch
import torch.nn as nn
import torchquantum as tq

__all__ = [
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "build_classifier_circuit",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "QuanvolutionHybridNet",
]

# --------------------------------------------------------------------------- #
# 1. Quantum filter
# --------------------------------------------------------------------------- #

class QuanvolutionFilter(tq.QuantumModule):
    """Hybrid patch‑wise quantum filter: classical 2×2 patch extraction followed by a variational kernel."""
    def __init__(
        self,
        patch_size: int = 2,
        stride: int = 2,
        n_qubits: int = 4,
        n_encodings: int = 8,
        n_layers: int = 2,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.n_qubits = n_qubits
        self.n_encodings = n_encodings
        self.n_layers = n_layers
        self._device = device or "cpu"

        # Classical conv for patch extraction
        self.conv = nn.Conv2d(1, 1, kernel_size=patch_size, stride=stride, bias=False)

        # Variational encoder
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_encodings, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=self._device)

        # Extract patches
        patches = self.conv(x).view(bsz, -1, self.patch_size * self.patch_size)
        patches = patches.reshape(bsz, patches.shape[1], self.patch_size * self.patch_size)

        # Encode each patch as rotation angles
        features = []
        for i in range(patches.shape[1]):
            data = patches[:, i, :]  # (B, 4)
            self.encoder(qdev, data)
            self.q_layer(qdev)
            meas = self.measure(qdev)  # (B, 4)
            features.append(meas.view(bsz, -1))
        return torch.cat(features, dim=1)

# --------------------------------------------------------------------------- #
# 2. Quantum classifier
# --------------------------------------------------------------------------- #

class QuanvolutionClassifier(nn.Module):
    """Quantum‑aware classifier: a quantum filter followed by a variational ansatz."""
    def __init__(
        self,
        patch_size: int = 2,
        stride: int = 2,
        n_qubits: int = 4,
        n_encodings: int = 8,
        n_layers: int = 2,
        classifier_depth: int = 2,
        num_classes: int = 10,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter(patch_size, stride, n_qubits, n_encodings, n_layers, device)
        # Build a quantum ansatz for classification
        self.classifier_circuit, _, _, _ = build_classifier_circuit(
            num_qubits=n_qubits * (28 // stride) * (28 // stride),
            depth=classifier_depth,
        )
        self.output = nn.Linear(
            self.filter.n_qubits * (28 // stride) * (28 // stride), num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter(x)
        logits = self.output(feats)
        return F.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------------- #
# 3. Quantum classifier factory
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[nn.Module, Iterable, Iterable, List]:
    """Construct a quantum ansatz for classification."""
    encoding = tq.ParameterVector("x", num_qubits)
    weights = tq.ParameterVector("theta", num_qubits * depth)

    circuit = tq.QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = []  # placeholder; can be expanded if needed
    return circuit, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
# 4. Quantum forward propagation utilities
# --------------------------------------------------------------------------- #

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return state.ptrace(keep)

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
    """Propagate quantum samples through a layered quantum network."""
    states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        states.append(layerwise)
    return states

# --------------------------------------------------------------------------- #
# 5. Fidelity utilities (quantum)
# --------------------------------------------------------------------------- #

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Absolute squared overlap of two pure states."""
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
# 6. Convenience wrapper
# --------------------------------------------------------------------------- #

class QuanvolutionHybridNet(nn.Module):
    """Convenience wrapper that bundles the quantum filter, classifier, and graph utilities."""
    def __init__(
        self,
        patch_size: int = 2,
        stride: int = 2,
        n_qubits: int = 4,
        n_encodings: int = 8,
        n_layers: int = 2,
        classifier_depth: int = 2,
        num_classes: int = 10,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter(
            patch_size, stride, n_qubits, n_encodings, n_layers, device
        )
        self.classifier_circuit, _, _, _ = build_classifier_circuit(
            num_qubits=n_qubits * (28 // stride) * (28 // stride),
            depth=classifier_depth,
        )
        self.output = nn.Linear(
            self.filter.n_qubits * (28 // stride) * (28 // stride), num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.filter(x)
        logits = self.output(x)
        return F.log_softmax(logits, dim=-1)
