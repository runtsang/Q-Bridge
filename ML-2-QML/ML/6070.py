"""Hybrid estimator that can evaluate classical PyTorch models, quantum circuits, or a hybrid combination of both.

The class exposes an ``evaluate`` method that accepts an iterable of observables and a sequence of parameter sets.
Observables can be either callables that accept a torch.Tensor and return a scalar (classical observables),
or Qiskit ``BaseOperator`` instances (quantum observables).  When both a classical model and a quantum circuit are
provided, the classical model produces a feature vector that is optionally transformed by a quantum head
before the final prediction.  Shot noise can be added to emulate measurement statistics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

# Optional import for quantum observables
try:
    from qiskit.quantum_info.operators.base_operator import BaseOperator
except Exception:
    BaseOperator = None  # type: ignore

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
QuantumObservable = BaseOperator

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D list of parameters into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridEstimator:
    """Hybrid estimator that can evaluate a classical PyTorch model, a quantum circuit, or both.

    Parameters
    ----------
    classical_model : nn.Module, optional
        A PyTorch module that produces feature vectors.  If omitted, no classical head is used.
    quantum_circuit : qiskit.circuit.QuantumCircuit, optional
        A Qiskit circuit that will be evaluated on a Statevector simulator (exact).  If omitted, no quantum part is used.
    quantum_head : nn.Module, optional
        A PyTorch module that maps the output of the quantum circuit to a final prediction.
        This is the quantum‑to‑classical bridge; it is optional and can be omitted for a purely quantum model.
    shots : int, optional
        Number of shots to add Gaussian shot noise to the expectation values.  If ``None`` the estimator is deterministic.
    seed : int, optional
        Random seed for the shot noise generator.
    """
    def __init__(
        self,
        *,
        classical_model: Optional[nn.Module] = None,
        quantum_circuit: Optional["qiskit.circuit.QuantumCircuit"] = None,
        quantum_head: Optional[nn.Module] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.classical_model = classical_model
        self.quantum_circuit = quantum_circuit
        self.quantum_head = quantum_head
        self.shots = shots
        self.seed = seed

        # Quick sanity checks
        if self.quantum_circuit is None and self.quantum_head is None:
            raise ValueError("Either a quantum circuit or a quantum head must be provided.")
        if self.classical_model is None and self.quantum_head is None:
            # Pure quantum model: fine
            pass

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[Union[ScalarObservable, QuantumObservable]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a list of observables for all parameter sets.

        Parameters
        ----------
        observables : iterable
            * For a classical model: a sequence of callables that return a scalar from a torch.Tensor.
            * For a quantum circuit: a list of Qiskit operators.
            * For a hybrid model: both callables and operators may be mixed; the order in the output row follows the
              order of ``observables``.
        parameter_sets : sequence
            Each element is a list/tuple of float parameters that will be fed to the model(s).

        Returns
        -------
        List[List[float]]
            A list of rows, each row contains the value of each observable for a single parameter set.
        """
        if not observables:
            raise ValueError("No observables provided.")
        obs = list(observables)

        results: List[List[float]] = []

        # --- Classical part ------------------------------------------
        if self.classical_model is not None:
            self.classical_model.eval()
            with torch.no_grad():
                for params in parameter_sets:
                    batch = _ensure_batch(params)
                    outputs = self.classical_model(batch)
                    # Apply quantum head if present
                    if self.quantum_head is not None:
                        outputs = self.quantum_head(outputs)
                    # Evaluate classical observables
                    row: List[float] = []
                    for observable in obs:
                        if callable(observable):
                            val = observable(outputs)
                            if isinstance(val, torch.Tensor):
                                scalar = float(val.mean().cpu())
                            else:
                                scalar = float(val)
                            row.append(scalar)
                        else:
                            # Placeholder for quantum observable; will be filled later
                            row.append(None)
                    results.append(row)

        # --- Quantum part --------------------------------------------
        if self.quantum_circuit is not None:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector

            def _bind(qc: QuantumCircuit, values: Sequence[float]) -> QuantumCircuit:
                if len(values)!= len(list(qc.parameters)):
                    raise ValueError("Parameter count mismatch for bound circuit.")
                mapping = dict(zip(list(qc.parameters), values))
                return qc.assign_parameters(mapping, inplace=False)

            for idx, params in enumerate(parameter_sets):
                state = Statevector.from_instruction(_bind(self.quantum_circuit, params))
                # Compute quantum observables
                if self.classical_model is not None:
                    # Replace None placeholders in the existing rows
                    for j, observable in enumerate(obs):
                        if isinstance(observable, BaseOperator):
                            val = state.expectation_value(observable)
                            results[idx][j] = float(val)
                else:
                    # Pure quantum case: build a new row with all quantum observables
                    row: List[float] = []
                    for observable in obs:
                        if isinstance(observable, BaseOperator):
                            val = state.expectation_value(observable)
                            row.append(float(val))
                    results.append(row)

        # --- Shot noise ------------------------------------------------
        if self.shots is not None and self.shots > 0:
            rng = np.random.default_rng(self.seed)
            noisy_results: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
                noisy_results.append(noisy_row)
            results = noisy_results

        return results

__all__ = ["HybridEstimator"]
