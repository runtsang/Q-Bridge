"""Quantum implementation of QuantumNATHybrid.

The architecture mirrors the classical version but replaces the
CNN encoder with a parametric GeneralEncoder and the graph
network with a variational quantum layer.  The module is a
torchquantum.QuantumModule, exposing the same forward API as
the classical counterpart.  A fidelity‑based adjacency graph
can be constructed from the measured Pauli‑Z expectation
states.  The evaluate method evaluates expectation values of
a list of observables for a batch of input parameters,
providing a quantum‑centric counterpart to the FastBaseEstimator.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import numpy as np


class QuantumNATHybrid(tq.QuantumModule):
    """Quantum hybrid model with a 4‑wire variational layer."""

    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    # -----------------------------------------------------------------------
    # Graph utilities
    # -----------------------------------------------------------------------
    def fidelity_graph(
        self,
        states: List[tq.QState],
        threshold: float = 0.9,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Construct a weighted graph from measured quantum states.
        Edges are weighted by the squared overlap (fidelity) between
        the states.  The construction mirrors the classical fidelity
        graph helper.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = abs((a.dag() * b)[0, 0]) ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # -----------------------------------------------------------------------
    # Estimator utilities
    # -----------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[tq.QOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Evaluate a list of quantum observables for a batch of
        parameter sets.  Each parameter set corresponds to a
        complete assignment of the circuit parameters.  The method
        constructs a bound circuit per set, obtains its statevector,
        and returns the expectation values.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            circ = self._to_circuit()
            circ.assign_parameters(dict(zip(circ.parameters, params)), inplace=True)
            state = tq.Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _to_circuit(self) -> tq.QuantumCircuit:
        """
        Convert the module into a stand‑alone circuit for evaluation.
        This is a lightweight helper that mirrors the forward
        construction without the measurement step.
        """
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1)
        self.encoder(qdev, torch.zeros(1, 16))
        self.q_layer(qdev)
        circ = qdev.get_circuit()
        circ.remove_final_measurements()
        return circ


__all__ = ["QuantumNATHybrid"]
