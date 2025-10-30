"""Quantum counterpart of the hybrid architecture, built with torchquantum.

The module name is QuantumNATHybrid to match the classical version.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import qutip as qt


class QuantumNATHybrid(tq.QuantumModule):
    """Quantum implementation of the hybrid Quantum‑NAT architecture."""

    def __init__(self, n_wires: int = 4, sampler_wires: int = 2) -> None:
        """
        Parameters
        ----------
        n_wires : int
            Number of wires used by the variational layer.
        sampler_wires : int
            Number of wires used for the sampler output.
        """
        super().__init__()
        self.n_wires = n_wires
        self.sampler_wires = sampler_wires

        # Encoder that maps classical data into a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])

        # Variational layer
        self.qlayer = self.QLayer(n_wires)

        # Measurement of all wires
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

        # Sampler parameters (used to rotate wire states before measurement)
        self.sampler_params = nn.Parameter(torch.randn(sampler_wires))

    # ----------------------------------------------------------------------
    class QLayer(tq.QuantumModule):
        """Variational block inspired by the original QFCModel."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that encodes classical data, applies a variational
        layer, and returns normalized measurement probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape [B, 1, H, W].

        Returns
        -------
        torch.Tensor
            Normalized measurement vector of shape [B, n_wires].
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Encode image into 4‑dim vector via average pooling
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.qlayer(qdev)

        # Optional sampler rotations before measurement
        if self.sampler_wires > 0:
            for i in range(self.sampler_wires):
                qdev.rx(self.sampler_params[i], i)

        out = self.measure(qdev)
        return self.norm(out)

    # ----------------------------------------------------------------------
    def fidelity_adjacency(
        self,
        states: Iterable[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Construct a weighted graph from qutip state fidelities.

        Parameters
        ----------
        states : Iterable[qt.Qobj]
            Iterable of state vectors.
        threshold : float
            Primary fidelity threshold for edge weight 1.0.
        secondary : float | None, optional
            Secondary threshold for edge weight ``secondary_weight``.
        secondary_weight : float, optional
            Weight assigned to secondary‑threshold edges.

        Returns
        -------
        nx.Graph
            Weighted graph where nodes correspond to states.
        """
        graph = nx.Graph()
        state_list = list(states)
        graph.add_nodes_from(range(len(state_list)))
        for i, a in enumerate(state_list):
            for j in range(i + 1, len(state_list)):
                b = state_list[j]
                fid = abs((a.dag() * b)[0, 0]) ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<QuantumNATHybrid n_wires={self.n_wires}>"
