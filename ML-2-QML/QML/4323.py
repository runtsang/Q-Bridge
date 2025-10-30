"""Hybrid classical–quantum model with CNN backbone and variational quantum layer.

This module defines :class:`QuantumNATHybrid` that merges the ideas from the three
reference pairs:

* The CNN + fully‑connected projection from the original
  :class:`QFCModel` (seed 1).
* A variational quantum circuit that accepts the pooled feature vector as
  input and produces a 4‑dimensional probability vector (seed 2).
* Graph‑based fidelity utilities from the third pair to allow the
  model to be trained on graph‑structured data or to construct a
  fidelity‑based regularisation term.

The architecture is intentionally *stateless* – all learnable
parameters are stored in ``self.parameters`` and can be optimised
with any PyTorch optimiser.  No training loop is provided; the
module is ready for integration into larger pipelines.

"""

from __future__ import annotations

import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import qutip as qt
import scipy as sc

# --------------------------------------------------------------------------- #
# Utility functions – fidelity, graph construction, random data
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
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    """Return a list of tuples (input, target) for supervised learning of a QNN."""
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    """Return a random network architecture and training data for a graph‑QNN."""
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


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
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


# --------------------------------------------------------------------------- #
# Classical CNN backbone – extracted from seed 1
# --------------------------------------------------------------------------- #

class _CNNBackbone(nn.Module):
    """A lightweight CNN that produces a 16‑dimensional feature vector."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# --------------------------------------------------------------------------- #
# Quantum variational layer – from seed 2
# --------------------------------------------------------------------------- #

class _VariationalLayer(tq.QuantumModule):
    """Parameterised quantum layer that acts on a 4‑wire device."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)


# --------------------------------------------------------------------------- #
# Hybrid model – classical backbone + quantum layer
# --------------------------------------------------------------------------- #

class QuantumNATHybrid(tq.QuantumModule):
    """Hybrid classical–quantum model that fuses a CNN backbone with a variational quantum layer.

    The forward pass performs:

    1. Classical feature extraction with a CNN.
    2. Feature pooling and encoding into a 4‑wire quantum device.
    3. Variational circuit execution.
    4. Measurement of all qubits and batch‑normalisation.

    The class also exposes graph‑based utilities for fidelity‑based adjacency
    construction, mirroring the API of the classical version.

    Examples
    --------
    >>> model = QuantumNATHybrid()
    >>> x = torch.randn(4, 1, 28, 28)
    >>> probs = model(x)
    >>> print(probs.shape)
    torch.Size([4, 4])
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = _VariationalLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining classical and quantum components."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Classical pre‑processing: pool features to a 16‑dim vector
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)

        # Encode the pooled features into the quantum device
        self.encoder(qdev, pooled)

        # Execute the variational circuit
        self.q_layer(qdev)

        # Measurement and normalisation
        out = self.measure(qdev)
        return self.norm(out)

    # --------------------------------------------------------------------- #
    # Graph utilities – expose the same API as in the classical version
    # --------------------------------------------------------------------- #

    @staticmethod
    def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Wrap the quantum‑based adjacency construction for compatibility."""
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                  secondary_weight=secondary_weight)

    @staticmethod
    def random_network(qnn_arch: list[int], samples: int):
        """Return a random graph‑QNN architecture and training data."""
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int):
        """Return random training data for a target unitary."""
        return random_training_data(unitary, samples)

__all__ = ["QuantumNATHybrid"]
