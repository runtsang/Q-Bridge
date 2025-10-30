"""GraphQNNGen397: quantum‑centric graph neural network.

This module provides a quantum‑driven interface that mirrors the
classical utilities while implementing forward propagation with
qutip operators.  It also exposes wrappers for a qiskit EstimatorQNN,
a torchquantum quantum LSTM, and a quanvolution filter, enabling
hybrid research workflows.
"""

from __future__ import annotations

import itertools
import networkx as nx
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import numpy as np
import qutip as qt
import torch
import torchquantum as tq
import torchquantum.functional as tqf

# Optional quantum helpers
try:
    from.EstimatorQNN import EstimatorQNN
except Exception:
    EstimatorQNN = None

try:
    from.QLSTM import QLSTM
except Exception:
    QLSTM = None

try:
    from.Conv import Conv
except Exception:
    Conv = None


class GraphQNNGen397:
    """Hybrid graph neural network that can run in quantum mode."""

    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(arch)):
            num_in = arch[layer - 1]
            num_out = arch[layer]
            ops: List[qt.Qobj] = []
            for out_idx in range(num_out):
                op = self._random_qubit_unitary(num_in + 1)
                if num_out > 1:
                    op = qt.tensor(
                        self._random_qubit_unitary(num_in + 1),
                        self._tensored_id(num_out - 1),
                    )
                    op = self._swap_registers(op, num_in, num_in + out_idx)
                ops.append(op)
            self.unitaries.append(ops)
        self._estimator = EstimatorQNN() if EstimatorQNN else None
        self._lstm_cls = QLSTM if QLSTM else None
        self._conv_cls = Conv if Conv else None

    @staticmethod
    def _tensored_id(num_qubits: int) -> qt.Qobj:
        """Identity operator on the specified number of qubits."""
        I = qt.qeye(2 ** num_qubits)
        dims = [2] * num_qubits
        I.dims = [dims.copy(), dims.copy()]
        return I

    @staticmethod
    def _tensored_zero(num_qubits: int) -> qt.Qobj:
        """Projector onto |0…0>."""
        zero = qt.fock(2 ** num_qubits).proj()
        dims = [2] * num_qubits
        zero.dims = [dims.copy(), dims.copy()]
        return zero

    @staticmethod
    def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        mat = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        U = np.linalg.qr(mat)[0]
        qobj = qt.Qobj(U)
        dims = [2] * num_qubits
        qobj.dims = [dims.copy(), dims.copy()]
        return qobj

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        vec = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
        vec /= np.linalg.norm(vec)
        state = qt.Qobj(vec)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    @classmethod
    def random_network(cls, arch: Sequence[int], samples: int) -> "GraphQNNGen397":
        """Instantiate a quantum network with random unitaries and training data."""
        instance = cls(arch)
        target_unitary = instance.unitaries[-1][0]
        training = []
        for _ in range(samples):
            st = cls._random_qubit_state(len(target_unitary.dims[0]))
            training.append((st, target_unitary * st))
        return instance, training

    def _partial_trace_keep(self, state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
        if len(keep)!= len(state.dims[0]):
            return state.ptrace(list(keep))
        return state

    def _partial_trace_remove(self, state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for idx in sorted(remove, reverse=True):
            keep.pop(idx)
        return self._partial_trace_keep(state, keep)

    def _layer_channel(
        self,
        layer: int,
        input_state: qt.Qobj,
    ) -> qt.Qobj:
        num_in = self.arch[layer - 1]
        num_out = self.arch[layer]
        state = qt.tensor(input_state, self._tensored_zero(num_out))
        layer_unitary = self.unitaries[layer][0].copy()
        for gate in self.unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary
        return self._partial_trace_remove(
            layer_unitary * state * layer_unitary.dag(), range(num_in)
        )

    def feedforward(self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        """Run each input through all layers and record intermediate states."""
        all_states: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current = sample
            for layer in range(1, len(self.arch)):
                current = self._layer_channel(layer, current)
                layerwise.append(current)
            all_states.append(layerwise)
        return all_states

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Squared magnitude of the overlap between two pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    def fidelity_adjacency(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(s_i, s_j)
            if fid >= threshold:
                g.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                g.add_edge(i, j, weight=secondary_weight)
        return g

    # Quantum‑aware wrappers ------------------------------------------------

    def run_estimator(self, data: qt.Qobj) -> qt.Qobj:
        if self._estimator is None:
            raise RuntimeError("EstimatorQNN module not available")
        return self._estimator

    def run_lstm(self, inputs: torch.Tensor, n_qubits: int = 0) -> torch.Tensor:
        if self._lstm_cls is None:
            raise RuntimeError("QLSTM module not available")
        lstm = self._lstm_cls(self.arch[0], self.arch[-1], n_qubits=n_qubits)
        output, _ = lstm(inputs)
        return output

    def run_conv(self, data: np.ndarray) -> float:
        if self._conv_cls is None:
            raise RuntimeError("Conv module not available")
        filt = self._conv_cls()
        return filt.run(data)


__all__ = ["GraphQNNGen397"]
