"""UnifiedQMLEstimator – a hybrid ML/QML estimator with PyTorch core and quantum extensions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
import networkx as nx
import itertools
import qutip as qt
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate a PyTorch model on batches of parameters and compute observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        row.append(float(val.mean().cpu()))
                    else:
                        row.append(float(val))
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Add optional Gaussian shot noise to emulate measurement statistics."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy

class QuantumVariationalLayer(nn.Module):
    """
    A lightweight variational circuit embedded in a PyTorch module.
    The circuit consists of a single qubit with a parameterized RX gate
    followed by a CZ entangling gate to a dummy qubit.  The expectation
    value of Z on the last qubit is returned as a scalar tensor.
    """
    def __init__(self, qubits: int = 2, params_per_layer: int = 1, layers: int = 1) -> None:
        super().__init__()
        self.qubits = qubits
        self.params_per_layer = params_per_layer
        self.layers = layers
        self.weight = nn.Parameter(torch.randn(layers * params_per_layer, dtype=torch.float32))
        # Build a reusable circuit template
        self._template = QuantumCircuit(qubits)
        self._params = []
        for layer in range(layers):
            for p in range(params_per_layer):
                param = qiskit.circuit.Parameter(f'theta_{layer}_{p}')
                self._params.append(param)
                self._template.ry(param, 0)
                if qubits > 1:
                    self._template.cz(0, 1)
            self._template.barrier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        circ = self._template.copy()
        param_map = {p: float(self.weight[i]) for i, p in enumerate(self._params)}
        circ.assign_parameters(param_map, inplace=True)
        state = Statevector.from_instruction(circ)
        dim = 2 ** self.qubits
        z_last = np.eye(dim, dtype=complex)
        for q in range(self.qubits - 1):
            z_last = np.kron(z_last, np.eye(2))
        z_last = np.kron(z_last, np.array([[1, 0], [0, -1]]))
        exp_val = float(np.vdot(state.data, z_last @ state.data).real)
        return torch.tensor(exp_val, dtype=torch.float32).unsqueeze(-1)

class UnifiedQMLEstimator(FastEstimator):
    """Hybrid estimator that can host a QuantumVariationalLayer inside a PyTorch model."""
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        return super().evaluate(observables, parameter_sets, shots=shots, seed=seed)

def _normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return t / (torch.norm(t) + 1e-12)

def state_fidelity(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """Fidelity between two classical activation vectors."""
    t1n = _normalize_tensor(t1)
    t2n = _normalize_tensor(t2)
    return float((t1n @ t2n).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, s1), (j, s2) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s1, s2)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

__all__ = [
    "UnifiedQMLEstimator",
    "FastBaseEstimator",
    "FastEstimator",
    "QuantumVariationalLayer",
    "state_fidelity",
    "fidelity_adjacency",
]
