"""Hybrid fraud‑detection module: classical backbone + photonic variational layer.

The module exposes two complementary classes:

* :class:`FraudLayerParameters` – immutable container for all layer hyper‑parameters.
* :class:`FraudDetectionHybridNet` – a :class:`torch.nn.Module` that builds a
  sequential network mirroring the photonic program and supports joint
  training. The forward pass returns both the classical output and the
  quantum expectation value, enabling hybrid loss functions.

The design pulls ideas from:

* The layer‑wise clipping and scaling logic of the original
  `FraudDetection.py` seed.
* The data‑generation utilities from the graph‑based `GraphQNN` seed to
  provide synthetic training pairs.
* The `EstimatorQNN` interface to expose a Qiskit estimator as an
  auxiliary quantum loss term.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
from torch import nn
import numpy as np
import networkx as nx
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

# --------------------------------------------------------------------------- #
# 1. Parameter container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FraudLayerParameters:
    """Immutable container for a single photonic layer's hyper‑parameters."""

    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# --------------------------------------------------------------------------- #
# 2. Helper functions
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _build_classical_layer(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Return a single classical layer that mimics a photonic module."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)

    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift

    return _Layer()

def _build_quantum_program(params: FraudLayerParameters, *, clip: bool) -> QuantumCircuit:
    """Return a Qiskit circuit that mirrors the photonic layer."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(2)
    # beam splitter
    qc.cx(0, 1)  # emulate BSgate with a CNOT
    qc.rz(theta, 0)
    qc.rz(phi, 1)
    # squeezing / displacement
    for i, (r, p) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qc.u3(r, p, 0) if i == 0 else qc.u3(r, p, 1)
    # second BS
    qc.cx(0, 1)
    # displacement / Kerr
    for i, (r, p) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qc.u3(r, p, 0) if i == 0 else qc.u3(r, 0, 1)
    for i, k in enumerate(params.kerr):
        qc.rx(2 * k, i)  # approximate Kerr via RX
    qc.measure_all()
    return qc

# --------------------------------------------------------------------------- #
# 3. Hybrid network
# --------------------------------------------------------------------------- #
class FraudDetectionHybridNet(nn.Module):
    """
    A hybrid neural network that combines a classical feed‑forward backbone
    with a photonic‑style variational circuit.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (classical) layer.
    hidden_layers : Iterable[FraudLayerParameters]
        Iterable of subsequent layer parameters.
    quantum_backbone : bool, optional
        If True, augment the classical network with a quantum circuit
        that outputs a single expectation value.  The circuit is
        instantiated once and its parameters are trainable via
        gradient‑based optimizers.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_layers: Iterable[FraudLayerParameters],
        quantum_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.classical = nn.Sequential(
            _build_classical_layer(input_params, clip=False),
            *(_build_classical_layer(lp, clip=True) for lp in hidden_layers),
            nn.Linear(2, 1),
        )
        self.quantum_backbone = quantum_backbone
        if quantum_backbone:
            # Build a single circuit that will be reused
            self.qc = _build_quantum_program(input_params, clip=False)
            # Wrap with a state‑vector estimator for back‑prop
            self.estimator = StatevectorEstimator(
                backend=qiskit.Aer.get_backend("statevector_simulator")
            )
            self.estimator_qnn = EstimatorQNN(
                circuit=self.qc,
                observables=[qiskit.circuit.quantum_info.SparsePauliOp.from_list([("Z", 1)])],
                input_params=[Parameter("theta")],
                weight_params=[Parameter("phi")],
                estimator=self.estimator,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        out_class : torch.Tensor
            The output of the classical sub‑network.
        out_q : torch.Tensor
            The expectation value from the backend‑agnostic quantum circuit.
        """
        out_class = self.classical(x)
        if self.quantum_backbone:
            # Evaluate quantum circuit; gradients flow through the estimator
            q_vals = self.estimator_qnn.predict(x)  # type: ignore[override]
            return out_class, q_vals
        else:
            return out_class, torch.zeros_like(out_class)

# --------------------------------------------------------------------------- #
# 4. Synthetic data utilities (borrowed from GraphQNN)
# --------------------------------------------------------------------------- #
def random_fraud_dataset(
    n_samples: int,
    arch: List[int],
    weight: torch.Tensor,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate random feature/target pairs using the same linear mapping
    used in :func:`GraphQNN.random_network`."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n_samples):
        features = torch.randn(weight.size(1))
        target = weight @ features
        dataset.append((features, target))
    return dataset

# --------------------------------------------------------------------------- #
# 5. Graph‑based adjacency for clustering fraud patterns
# --------------------------------------------------------------------------- #
def fraud_graph_from_fidelities(
    states: List[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Construct a graph where nodes are data points and edges encode
    fidelity‑like similarity (here we use cosine similarity of tensors)."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j, b in enumerate(states[i + 1 :], start=i + 1):
            cos = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-12)
            fid = cos.item() ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionHybridNet",
    "random_fraud_dataset",
    "fraud_graph_from_fidelities",
]
