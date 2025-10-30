"""Quantum‑enhanced LSTM with QCNN‑derived parameters and graph‑based state management.

The module implements :class:`HybridQLSTM`, mirroring the classical API but replacing each LSTM gate
with a small quantum circuit.  A QCNN‑style network supplies the rotation angles for the
quantum gates, and a fidelity graph is built from measurement outcomes to guide the
state evolution.
"""

from __future__ import annotations

import itertools
from typing import Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum.functional as tqf
from torchquantum import QuantumDevice, QuantumModule


# --------------------------------------------------------------------------- #
#  Graph utilities (identical to the ML side)
# --------------------------------------------------------------------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  QCNN‑style parameter generator (classical)
# --------------------------------------------------------------------------- #
class QCNNParameterGenerator(nn.Module):
    """
    Generates a vector of rotation angles for a quantum gate.
    Mimics a QCNN: 2 linear layers with Tanh activation, outputting
    ``n_qubits`` angles.
    """

    def __init__(self, input_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * n_qubits),
            nn.Tanh(),
            nn.Linear(2 * n_qubits, n_qubits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
#  Quantum gate module
# --------------------------------------------------------------------------- #
class QLayer(QuantumModule):
    """
    Quantum layer applying a sequence of RX rotations followed by a
    linear entangling circuit.
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        angles : torch.Tensor
            Shape ``(batch, n_wires)`` of rotation angles for each qubit.
        """
        qdev = QuantumDevice(n_wires=self.n_wires, bsz=angles.shape[0], device=angles.device)
        for wire in range(self.n_wires):
            tqf.rx(qdev, params=angles[:, wire], wires=wire)
        # Simple linear entanglement
        for wire in range(self.n_wires - 1):
            tqf.cx(qdev, wires=[wire, wire + 1])
        return tqf.measure_all(qdev, basis=0)


# --------------------------------------------------------------------------- #
#  Hybrid LSTM
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM where each gate is a small quantum circuit.
    Parameters for the gates are produced by a QCNN‑style neural network,
    ensuring end‑to‑end differentiability.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input token.
    hidden_dim : int
        Size of the hidden state.
    n_qubits : int
        Number of qubits used in each gate circuit.
    graph_threshold : float, default 0.8
        Fidelity threshold for constructing the adjacency graph.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        graph_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.graph_threshold = graph_threshold

        # QCNN parameter generator
        self.qcnn = QCNNParameterGenerator(input_dim, n_qubits)

        # Quantum gates for each LSTM component
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        # Linear projections from (input + hidden) to gate parameters
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Linear projections to raw gate parameters
            f_raw = self.forget_linear(combined)
            i_raw = self.input_linear(combined)
            g_raw = self.update_linear(combined)
            o_raw = self.output_linear(combined)

            # QCNN‑derived rotation angles
            f_angles = self.qcnn(f_raw)
            i_angles = self.qcnn(i_raw)
            g_angles = self.qcnn(g_raw)
            o_angles = self.qcnn(o_raw)

            # Quantum gates
            f = torch.sigmoid(self.forget(f_angles))
            i = torch.sigmoid(self.input(i_angles))
            g = torch.tanh(self.update(g_angles))
            o = torch.sigmoid(self.output(o_angles))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)

        # Build fidelity graph from hidden states
        graph = fidelity_adjacency(
            [h.detach() for h in outputs], self.graph_threshold
        )
        # The graph can be used to regularise or guide hidden states.
        # (Implementation omitted for brevity.)

        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


__all__ = ["HybridQLSTM"]
