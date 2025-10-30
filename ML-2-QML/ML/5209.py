"""Hybrid estimator blending classical neural networks and quantum circuits.

This module extends the original FastBaseEstimator to support both
pure‑classical PyTorch models and hybrid models that embed a quantum
circuit as a final head.  The interface remains identical to the seed
so downstream pipelines can swap in the new implementation without
modification.

Key extensions:
- `HybridHead` – a small nn.Module that forwards activations through
  a parameterised quantum circuit and returns the expectation value
  of Pauli‑Z.  This mirrors the `Hybrid` class in the
  ClassicalQuantumBinaryClassification example.
- `FastHybridEstimator` – accepts either a plain ``nn.Module`` or a
  ``HybridHead`` instance.  It automatically routes the parameters
  to the appropriate sub‑module and supports optional Gaussian shot
  noise, just like the original `FastEstimator`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# --------------------------------------------------------------------------- #
# Quantum head for hybrid models
# --------------------------------------------------------------------------- #
class HybridHead(nn.Module):
    """
    A lightweight head that forwards a scalar activation through a
    parameterised quantum circuit and returns the expectation of
    Pauli‑Z.  The circuit is implemented with qiskit and executed
    on a simulator.  It is inspired by the `Hybrid` class from the
    ClassicalQuantumBinaryClassification example.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    backend : qiskit.providers.Backend
        The backend used to simulate the circuit.
    shots : int
        Number of shots per evaluation.
    shift : float
        Parameter shift used for gradient estimation.
    """

    def __init__(
        self,
        n_qubits: int,
        backend,
        shots: int = 100,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift

        # Build a simple Ry‑parameterised circuit.
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        qc.ry(self.shift, range(self.n_qubits))
        qc.measure_all()
        return qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward a scalar or 1‑D tensor through the circuit and return the
        expectation value of Pauli‑Z.
        """
        # Convert tensor to numpy for Qiskit.
        thetas = inputs.detach().cpu().numpy()
        if thetas.ndim == 0:
            thetas = thetas.reshape(1)
        # Transpile and assemble
        from qiskit import transpile, assemble

        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self._circuit.parameters[0]: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return torch.tensor([expectation(r) for r in result], dtype=torch.float32)
        return torch.tensor([expectation(result)], dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Unified estimator
# --------------------------------------------------------------------------- #
class FastHybridEstimator:
    """
    Evaluates either a pure classical neural network or a hybrid model that
    ends with a ``HybridHead``.  The API is identical to the original
    FastBaseEstimator, so existing code can instantiate this class
    without changes.

    Parameters
    ----------
    model : nn.Module
        Either a plain ``nn.Module`` or a ``HybridHead`` instance.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set.  If ``shots`` is
        provided, Gaussian shot noise is added to mimic quantum
        measurement statistics.

        Parameters
        ----------
        observables : iterable
            Callables that map a model output tensor to a scalar or
            a list of scalars.
        parameter_sets : sequence
            Sequence of parameter vectors to evaluate.
        shots : int, optional
            Number of shots for stochastic noise.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        List[List[float]]
            Nested list where each inner list contains the observable
            values for a single parameter set.
        """
        # Default observable is mean of outputs.
        observables = list(observables) or [lambda o: o.mean(dim=-1)]

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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                rng.normal(mean, max(1e-6, 1 / shots)) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastHybridEstimator", "HybridHead"]
