"""Unified quantum‑classical estimator with graph‑based fidelity and self‑attention.

This module defines ``UnifiedQuantumEstimator`` that combines the
FastBaseEstimator, GraphQNN, SelfAttention, and QTransformerTorch
concepts into a single, extensible API.

The estimator accepts a PyTorch ``nn.Module`` as the classical backbone
and an optional quantum circuit that implements the same ``evaluate``
signature.  It can optionally wrap the model output with a self‑attention
block (classical or quantum) and provides utilities for building a
fidelity‑based graph from hidden states.

The design is deliberately lightweight so that it can be dropped into
existing codebases without heavy dependencies beyond PyTorch,
NumPy, and NetworkX.  Quantum functionality is optional and uses
Qiskit only when a quantum circuit is supplied.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor with a leading batch dimension."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class UnifiedQuantumEstimator(nn.Module):
    """Hybrid estimator that fuses FastBaseEstimator, GraphQNN, and SelfAttention.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch network.  The model is expected to output a tensor of
        shape ``(batch, features)``.  Hidden states can be accessed via the
        ``hidden`` attribute if the model implements it; otherwise the
        final output is used.
    quantum_circuit : Optional[Any]
        An object that implements ``evaluate(observables, parameter_sets)``
        and returns a list of lists.  It is used only when the ``use_quantum``
        flag is set.
    use_quantum : bool
        When ``True`` the estimator delegates to the quantum circuit;
        otherwise the classical model is used.
    attention : Optional[nn.Module]
        A self‑attention block.  It can be a classical
        ``nn.MultiheadAttention`` or a quantum variant defined below.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        quantum_circuit: Optional[Any] = None,
        use_quantum: bool = False,
        attention: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.quantum_circuit = quantum_circuit
        self.use_quantum = use_quantum
        self.attention = attention

    # ------------------------------------------------------------------ #
    #  Classical evaluation path
    # ------------------------------------------------------------------ #
    def _evaluate_classical(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Run the PyTorch model and compute observables."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                # If the model exposes hidden states, gather them.
                hidden = getattr(self.model, "hidden", outputs)
                # Apply optional attention
                if self.attention is not None:
                    hidden = self.attention(hidden)
                row: List[float] = []
                for observable in observables:
                    value = observable(hidden)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

    # ------------------------------------------------------------------ #
    #  Quantum evaluation path
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Public API that forwards to the classical or quantum evaluator."""
        if self.use_quantum and self.quantum_circuit is not None:
            return self.quantum_circuit.evaluate(observables, parameter_sets)
        return self._evaluate_classical(observables, parameter_sets)

    # ------------------------------------------------------------------ #
    #  Graph‑based fidelity adjacency
    # ------------------------------------------------------------------ #
    def build_graph(
        self,
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a NetworkX graph built from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, s_i in enumerate(states):
            for j, s_j in enumerate(states[i + 1 :], start=i + 1):
                fid = self._state_fidelity(s_i, s_j)
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def _state_fidelity(state_a: torch.Tensor, state_b: torch.Tensor) -> float:
        """Cosine‑like fidelity between two 1‑D tensors."""
        a_norm = state_a / (torch.norm(state_a) + 1e-12)
        b_norm = state_b / (torch.norm(state_b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    # ------------------------------------------------------------------ #
    #  Utility for generating random training data
    # ------------------------------------------------------------------ #
    def random_training_data(
        self, weight: torch.Tensor, samples: int
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """Return a sample set for a linear‑layer training task."""
        dataset: List[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    # ------------------------------------------------------------------ #
    #  Self‑attention wrapper for the classical model
    # ------------------------------------------------------------------ #
    def self_attention(
        self,
        embed_dim: int,
        *,
        use_quantum: bool = False,
        n_qubits: int = 4,
    ) -> nn.Module:
        """Return a self‑attention module that can be used as ``attention``."""
        if not use_quantum:
            # Classical multi‑head attention
            return nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        # Quantum self‑attention implemented with Qiskit
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
        import numpy as np

        class QuantumSelfAttention(nn.Module):
            def __init__(self, n_qubits: int):
                super().__init__()
                self.n_qubits = n_qubits
                self.qr = QuantumRegister(n_qubits, "q")
                self.cr = ClassicalRegister(n_qubits, "c")
                self.backend = Aer.get_backend("qasm_simulator")

            def _build_circuit(
                self, rotation_params: np.ndarray, entangle_params: np.ndarray
            ) -> QuantumCircuit:
                circuit = QuantumCircuit(self.qr, self.cr)
                for i in range(self.n_qubits):
                    circuit.rx(rotation_params[3 * i], i)
                    circuit.ry(rotation_params[3 * i + 1], i)
                    circuit.rz(rotation_params[3 * i + 2], i)
                for i in range(self.n_qubits - 1):
                    circuit.crx(entangle_params[i], i, i + 1)
                circuit.measure(self.qr, self.cr)
                return circuit

            def forward(
                self,
                inputs: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor,
                shots: int = 1024,
            ) -> torch.Tensor:
                # Flatten inputs to a 1‑D array of floats
                flat = inputs.flatten().cpu().numpy()
                circuit = self._build_circuit(flat, flat)
                job = execute(circuit, self.backend, shots=shots)
                counts = job.result().get_counts(circuit)
                # Convert counts to a probability distribution
                probs = np.array([counts.get(f"{i:0{self.n_qubits}b}", 0) for i in range(2 ** self.n_qubits)])
                probs /= probs.sum()
                return torch.tensor(probs, dtype=torch.float32, device=inputs.device)

        return QuantumSelfAttention(n_qubits)

    # ------------------------------------------------------------------ #
    #  Forward method (optional convenience)
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that forwards to the underlying model."""
        return self.model(x)
