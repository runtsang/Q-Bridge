"""Hybrid quantum graph neural network with optional quantum quanvolution filter.

The class `HybridGraphQNN` operates on pure qubit states using
qutip.  It optionally prefixes the input with a
`QuanvolutionFilter` that implements a small variational circuit on
image patches.  The resulting classical features are amplitude‑encoded
into a 10‑qubit state before entering the variational layers.

The public API matches that of the classical module, making it
straightforward to swap between regimes.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qutip as qt
import torchquantum as tq
import torch

# --------------------------------------------------------------------------- #
#  Helper utilities for qubit operations
# --------------------------------------------------------------------------- #

def _tensored_identity(num_qubits: int) -> qt.Qobj:
    I = qt.qeye(2**num_qubits)
    dims = [2] * num_qubits
    I.dims = [dims.copy(), dims.copy()]
    return I

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    proj = qt.fock(2**num_qubits).proj()
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
    dim = 2**num_qubits
    M = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    U, _ = np.linalg.qr(M)  # orthogonal matrix
    qobj = qt.Qobj(U)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2**num_qubits
    vec = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    vec /= np.linalg.norm(vec)
    state = qt.Qobj(vec)
    state.dims = [[2]*num_qubits, [1]*num_qubits]
    return state

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
    unitaries: Sequence[List[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    """Apply a variational layer and trace out ancilla qubits."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

# --------------------------------------------------------------------------- #
#  Quantum quanvolution filter
# --------------------------------------------------------------------------- #

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum kernel applied to 2×2 image patches."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # ``x`` is a flat vector → reshape to 28×28 image
        img = x.view(bsz, 28, 28)
        patches: List[torch.Tensor] = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        img[:, r, c],
                        img[:, r, c + 1],
                        img[:, r + 1, c],
                        img[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))

        return torch.cat(patches, dim=1)  # (bsz, 4 * 14 * 14)

# --------------------------------------------------------------------------- #
#  Hybrid quantum graph‑neural‑network class
# --------------------------------------------------------------------------- #

class HybridGraphQNN:
    """Quantum GNN optionally preceded by a quantum quanvolution filter.

    Parameters
    ----------
    qnn_arch:
        Sequence of hidden layer widths (input → … → output).
    use_quanvolution:
        If ``True`` the input is first processed by
        :class:`~QuanvolutionFilter`.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        use_quanvolution: bool = False,
    ) -> None:
        self.arch = list(qnn_arch)
        self.use_quanvolution = use_quanvolution
        self.unitaries: List[List[qt.Qobj]] = [[]]

        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            layer_ops: List[qt.Qobj] = []

            for out_idx in range(num_outputs):
                op = _random_qubit_unitary(num_inputs + 1)  # +1 for ancilla
                if num_outputs > 1:
                    op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_identity(num_outputs - 1))
                    op = _swap_registers(op, num_inputs, num_inputs + out_idx)
                layer_ops.append(op)

            self.unitaries.append(layer_ops)

        self.qfilter: QuanvolutionFilter | None = (
            QuanvolutionFilter() if use_quanvolution else None
        )

    # --------------------------------------------------------------------- #
    #  Random data / unitary generators
    # --------------------------------------------------------------------- #

    @staticmethod
    def random_network(
        qnn_arch: Sequence[int],
        samples: int,
    ) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        """Create a random network and training data."""
        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs, num_outputs = qnn_arch[layer - 1], qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []

            for out_idx in range(num_outputs):
                op = _random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_identity(num_outputs - 1))
                    op = _swap_registers(op, num_inputs, num_inputs + out_idx)
                layer_ops.append(op)

            unitaries.append(layer_ops)

        target_unitary = _random_qubit_unitary(qnn_arch[-1])
        training_data = HybridGraphQNN.random_training_data(target_unitary, samples)
        return list(qnn_arch), unitaries, training_data, target_unitary

    @staticmethod
    def random_training_data(
        unitary: qt.Qobj,
        samples: int,
    ) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        """Generate samples ``(state, unitary * state)``."""
        dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = _random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    # --------------------------------------------------------------------- #
    #  Forward propagation
    # --------------------------------------------------------------------- #

    def feedforward(
        self,
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        """Run the quantum network on a batch of samples."""
        outputs: List[List[qt.Qobj]] = []

        for state, _ in samples:
            current_state = state

            # Optional quanvolution preprocessing
            if self.use_quanvolution and self.qfilter is not None:
                # Convert the state vector to a Torch tensor
                vec = current_state.full().reshape(-1)
                tensor = torch.from_numpy(vec).float().unsqueeze(0)  # (1, 2**n)
                filtered = self.qfilter(tensor)  # (1, 4*14*14)
                # Amplitude encode the classical vector into a 10‑qubit state
                amp_vec = filtered.squeeze(0).numpy()
                encoded = self._amplitude_encode(amp_vec, num_qubits=10)
                current_state = encoded

            activations: List[qt.Qobj] = [current_state]
            for layer, layer_ops in enumerate(self.unitaries[1:], start=1):
                current_state = _layer_channel(
                    self.arch,
                    self.unitaries,
                    layer,
                    current_state,
                )
                activations.append(current_state)

            outputs.append(activations)

        return outputs

    @staticmethod
    def _amplitude_encode(vector: np.ndarray, num_qubits: int) -> qt.Qobj:
        """Encode a classical vector into a pure quantum state."""
        dim = 2**num_qubits
        padded = np.zeros((dim, 1), dtype=np.complex128)
        length = min(len(vector), dim)
        padded[:length, 0] = vector[:length]
        norm = np.linalg.norm(padded)
        if norm > 0:
            padded /= norm
        state = qt.Qobj(padded)
        state.dims = [[2]*num_qubits, [1]*num_qubits]
        return state

    # --------------------------------------------------------------------- #
    #  Utility functions
    # --------------------------------------------------------------------- #

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Squared overlap between two pure states."""
        return float(abs((a.dag() * b)[0, 0]) ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from pairwise fidelities of ``states``."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = [
    "QuanvolutionFilter",
    "HybridGraphQNN",
]
