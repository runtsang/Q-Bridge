"""FraudGraphHybrid – classical core + quantum similarity layer.

The module exposes:
- FraudLayerParameters: holds a single layer’s hyper‑parameters.
- _layer_from_params: builds one linear‑Tanh‑scale block.
- build_fraud_detection_program: stacks many layers and a final linear head.
- quantum_similarity_matrix: evaluates a Qiskit variational circuit for each input and returns the state‑vector fidelity matrix.
- fidelity_adjacency: constructs a weighted graph from the fidelity matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import torch
from torch import nn
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
import networkx as nx

# --------------------------------------------------------------------------- #
# 1. Classical fraud‑detection core
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters that mimic a single photonic layer but implemented classically."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    """Bound a parameter to keep the training space compact."""
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Return a linear‑Tanh‑scale block parameterised by `params`."""
    weight = torch.tensor([[params.bs_theta, params.bs_phi]], dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 1)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(x))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the photonic architecture."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(1, 1))  # final fraud‑score head
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 2. Quantum similarity layer
# --------------------------------------------------------------------------- #

def _variational_circuit(num_qubits: int, params: np.ndarray) -> QuantumCircuit:
    """Return a RealAmplitudes circuit parameterised by `params`."""
    qc = RealAmplitudes(num_qubits, reps=2, entanglement="circular")
    qc.assign_parameters(params, inplace=True)
    return qc

def quantum_similarity_matrix(
    inputs: torch.Tensor,
    qubit_count: int = 2,
    seed: int | None = None,
) -> np.ndarray:
    """
    For each input sample, build a variational circuit
    and compute its state‑vector.  Return the fidelity matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    batch = inputs.detach().cpu().numpy()
    n_samples = batch.shape[0]
    statevectors: List[Statevector] = []

    for vec in batch:
        # encode the classical vector into circuit parameters
        param_count = 2 * qubit_count
        params = np.linspace(0, 2 * np.pi, param_count) * vec[0]  # simple mapping
        qc = _variational_circuit(qubit_count, params)
        sv = Statevector.from_instruction(qc)
        statevectors.append(sv)

    # compute pairwise fidelities
    fid_matrix = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples):
        for j in range(i, n_samples):
            fid = abs(statevectors[i].data.conj().dot(statevectors[j].data)) ** 2
            fid_matrix[i, j] = fid
            fid_matrix[j, i] = fid
    return fid_matrix

def fidelity_adjacency(
    fidelity_matrix: np.ndarray,
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """
    Build a weighted graph from a fidelity matrix.
    Edges with fidelity ≥ `threshold` get weight 1.
    Optionally add weaker edges between `secondary` and `threshold`.
    """
    G = nx.Graph()
    G.add_nodes_from(range(fidelity_matrix.shape[0]))
    for i in range(fidelity_matrix.shape[0]):
        for j in range(i + 1, fidelity_matrix.shape[0]):
            fid = fidelity_matrix[i, j]
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
# 3. Utilities for synthetic data
# --------------------------------------------------------------------------- #

def random_training_data(samples: int) -> List[tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate synthetic transaction pairs:
    - features: random 2‑D vector
    - target: a linear transformation of the features
    """
    data = []
    weight = torch.randn(1, 2)
    for _ in range(samples):
        features = torch.randn(2)
        target = weight @ features
        data.append((features, target))
    return data
