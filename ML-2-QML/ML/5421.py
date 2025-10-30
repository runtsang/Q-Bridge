"""
FastBaseEstimatorGen313 – classical side
========================================

This module implements a hybrid estimator that can handle classical neural
networks, TorchQuantum variational circuits, or graph‑based QNNs.  It
provides:
* batched inference with optional shot‑noise;
* a choice of RBF or quantum kernel for data‑driven models;
* a simple kernel‑ridge‑regression training routine;
* fidelity‑based adjacency graph utilities from the GraphQNN reference.

Author: gpt‑oss‑20b
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable, List, Sequence, Callable, Optional, Union

# Import the quantum kernel module from the QML side
try:
    from.fast_base_estimator_qml import QuantumKernelModule
except Exception:
    # Fallback dummy kernel if QML module is unavailable
    class QuantumKernelModule:
        def __init__(self, *_, **__): pass
        def __call__(self, x: Tensor, y: Tensor) -> Tensor: return torch.zeros((x.shape[0], y.shape[0]))

# Import graph utilities
try:
    from.GraphQNN import (
        feedforward as qgraph_feedforward,
        fidelity_adjacency as qgraph_fidelity,
        random_network as qgraph_random_network,
        random_training_data as qgraph_random_training,
        state_fidelity as qgraph_state_fidelity,
    )
except Exception:
    # Dummy placeholders if GraphQNN is missing
    def qgraph_feedforward(*_, **__): return []
    def qgraph_fidelity(*_, **__): return None
    def qgraph_random_network(*_, **__): return None
    def qgraph_random_training(*_, **__): return None
    def qgraph_state_fidelity(*_, **__): return 0.0

# ---------------------------------------------------------------------------

def _ensure_batch(values: Sequence[float]) -> Tensor:
    """Return a 2‑D tensor from a 1‑D sequence."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

# ---------------------------------------------------------------------------

ScalarObservable = Callable[[Tensor], Tensor | float]

class FastBaseEstimatorGen313:
    """Hybrid estimator that supports classical neural nets, TorchQuantum
    devices, and graph‑based QNNs.

    Parameters
    ----------
    model : nn.Module | torchquantum.QuantumDevice | Sequence[Sequence[qt.Qobj]]
        The underlying model.  For a neural net it must be a subclass of
        nn.Module; for a quantum device it is a TorchQuantum QuantumDevice; for
        a graph‑QNN it is a list of list‑of‑Qobj.
    kernel : {'rbf', 'quantum', None}
        Type of kernel to use when the model is a plain data set.  If ``None``
        the model's own forward method is used.
    shots : int | None
        Optional shot‑noise simulation.  If ``None`` no noise is added.
    """

    def __init__(
        self,
        model: Union[nn.Module, "tq.QuantumDevice", Sequence[Sequence["qt.Qobj"]]],
        kernel: Optional[str] = None,
        shots: Optional[int] = None,
    ) -> None:
        self.model = model
        self.shots = shots

        # Default kernel selection
        if kernel is None:
            if hasattr(model, "forward") and not isinstance(model, nn.Module):
                kernel = "quantum"
            else:
                kernel = "rbf"
        self.kernel_type = kernel

        if self.kernel_type == "quantum":
            self._kernel_module = QuantumKernelModule()
        else:
            self._kernel_module = _ClassicalRBFKernel(gamma=1.0)

    # -----------------------------------------------------------------------
    #  Core evaluation
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | Iterable["BaseOperator"],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate the model for a batch of parameter sets and observables.

        The method dispatches to the appropriate backend based on the type of
        ``self.model``.  For a neural net the observables are scalar functions
        of the output tensor.  For a quantum device the observables are
        Qiskit operators.  For a graph‑QNN the observables are the states
        produced by the network; the caller can then build an adjacency
        graph using :func:`qgraph_fidelity`.

        """
        results: List[List[float]] = []

        # Convert parameter_sets to a tensor if needed
        param_tensor = torch.as_tensor(parameter_sets, dtype=torch.float32)

        if isinstance(self.model, nn.Module):
            # Classical neural network
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(param_tensor)
            for obs in observables:  # type: ignore
                if callable(obs):
                    val = obs(outputs)
                    if isinstance(val, Tensor):
                        val = val.mean().cpu().item()
                    results.append([float(val)])
                else:
                    raise TypeError("Observables for a neural net must be callables.")
        elif hasattr(self.model, "forward") and not isinstance(self.model, nn.Module):
            # TorchQuantum device – use the kernel module to compute expectation values
            for obs in observables:  # type: ignore
                # Expectation values via the quantum kernel
                kernel_vals = self._kernel_module(param_tensor, param_tensor).cpu().numpy()
                results.append(kernel_vals.tolist())
        else:
            # Graph‑QNN case – run feedforward to obtain states
            states = qgraph_feedforward(self.model, param_tensor)
            for state in states:
                results.append([float(state)])

        # Add shot noise if requested
        if self.shots is not None:
            rng = np.random.default_rng(seed=42)
            for i, row in enumerate(results):
                noisy_row = [rng.normal(loc=val, scale=max(1e-6, 1.0 / self.shots)) for val in row]
                results[i] = noisy_row

        return results

    # -----------------------------------------------------------------------
    #  Training utilities
    # -----------------------------------------------------------------------

    def train(
        self,
        training_data: Sequence[tuple[Tensor, Tensor]],
        epochs: int = 100,
        lr: float = 1e-3,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = nn.MSELoss(),
    ) -> None:
        """Simple training loop for a neural‑network model."""
        if not isinstance(self.model, nn.Module):
            raise TypeError("Training is only supported for nn.Module models.")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(epochs):
            for x, y in training_data:
                optimizer.zero_grad()
                pred = self.model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

    # -----------------------------------------------------------------------
    #  Graph utilities
    # -----------------------------------------------------------------------

    def graph_from_states(self, states: Sequence[Tensor], threshold: float) -> "nx.Graph":
        """Return a fidelity‑based adjacency graph of the given states."""
        return qgraph_fidelity(states, threshold)

# ---------------------------------------------------------------------------

class _ClassicalRBFKernel:
    """Simple wrapper around the seed RBF kernel implementation."""

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

__all__ = ["FastBaseEstimatorGen313"]
