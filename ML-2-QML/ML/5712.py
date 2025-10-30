"""Unified estimator that can evaluate classical neural networks or quantum circuits.

The estimator keeps the original FastBaseEstimator API but adds a
`backend` keyword that determines which backend is used.  The
`evaluate` method dispatches to either a PyTorch or a Qiskit
implementation.  The implementation is intentionally lightweight so
that the estimator can be used as a drop‑in replacement for the
original FastBaseEstimator in either repository.

Note:
    * The deterministic part is implemented with PyTorch and mirrors the
      original FastBaseEstimator.
    * The quantum part uses Qiskit to compute expectation values on a
      Statevector.  It can be extended to a real quantum device if
      desired.
    * The noise injection logic is shared: Gaussian noise for
      deterministic runs, shot‑noise for quantum runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Any, Optional

import numpy as np
import torch
from torch import nn

# Type alias for a scalar observable on a torch tensor
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class UnifiedGraphEstimator:
    """Estimator that evaluates either a classical model or a quantum circuit.

    Parameters
    ----------
    model : nn.Module | QuantumCircuit
        A PyTorch model or a Qiskit QuantumCircuit.  The estimator
        automatically detects the type and uses the appropriate
        backend.
    shots : int | None, optional
        Number of shots for quantum evaluation or Gaussian noise level
        for classical evaluation.  ``None`` means noiseless.
    seed : int | None, optional
        Random seed for reproducibility.
    """

    def __init__(self, model: Any, *, shots: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Detect backend
        try:
            from qiskit import QuantumCircuit
            self._is_quantum = isinstance(model, QuantumCircuit)
        except Exception:
            self._is_quantum = False

        if self._is_quantum:
            from qiskit.quantum_info import Statevector
            self._Statevector = Statevector
        else:
            self._Statevector = None

    def _evaluate_pytorch(self, observables: Iterable[ScalarObservable],
                          parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Deterministic PyTorch evaluation."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
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
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def _evaluate_qiskit(self, observables: Iterable[Any],
                         parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Expectation evaluation on a Statevector."""
        from qiskit.quantum_info import Statevector
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self.model.assign_parameters(dict(zip(self.model.parameters, params)), inplace=False)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate(self, observables: Iterable[Any],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Dispatch to the appropriate backend and optionally add noise."""
        if self._is_quantum:
            raw = self._evaluate_qiskit(observables, parameter_sets)
            if self.shots is None:
                return [[float(v) for v in row] for row in raw]
            # Shot noise: sample from normal distribution around expectation
            noisy = []
            for row in raw:
                noisy_row = [float(self._rng.normal(float(v), max(1e-6, 1 / self.shots))) for v in row]
                noisy.append(noisy_row)
            return noisy
        else:
            raw = self._evaluate_pytorch(observables, parameter_sets)
            if self.shots is None:
                return raw
            noisy = []
            for row in raw:
                noisy_row = [float(self._rng.normal(val, max(1e-6, 1 / self.shots))) for val in row]
                noisy.append(noisy_row)
            return noisy

    # ------------------------------------------------------------------
    # Graph utilities – shared between classical and quantum regimes
    # ------------------------------------------------------------------
    @staticmethod
    def fidelity(a: Any, b: Any) -> float:
        """Return the fidelity between two states.

        For classical tensors the fidelity is the squared dot product of
        L2‑normalized vectors.  For Qiskit `Statevector` the overlap is
        computed using the inner product.
        """
        if hasattr(a, "norm"):
            a_norm = a / (a.norm() + 1e-12)
            b_norm = b / (b.norm() + 1e-12)
            return float((a_norm @ b_norm).item() ** 2)
        # Assume Qiskit Statevector
        return float(abs(np.vdot(a.data, b.data)) ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Any], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> Any:
        """Build a weighted graph from state fidelities.

        The function accepts a list of either torch tensors or quantum
        states.  It returns a `networkx.Graph` with edges weighted
        according to the fidelity thresholds.
        """
        import networkx as nx
        import itertools

        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = UnifiedGraphEstimator.fidelity(s_i, s_j)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G
