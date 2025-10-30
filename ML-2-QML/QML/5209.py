"""Quantum‑centric estimator that evaluates quantum models or circuits.

The implementation mirrors the classical version but operates on
QuantumModule instances from torchquantum or on Qiskit circuits.
It can also wrap a hybrid CNN‑plus‑quantum‑head model inspired by
the ClassicalQuantumBinaryClassification example.

Key features
------------
- Supports pure quantum modules (e.g. `QModel` from the regression
  example) and hybrid models that combine a classical CNN with a
  quantum head.
- The `evaluate` method accepts observables as either
  `torchquantum` operators or callables that operate on the output
  tensor.
- Optional shot‑noise simulation is handled internally by the
  quantum backend.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Quantum head for hybrid models (torchquantum based)
# --------------------------------------------------------------------------- #
class HybridQuantumHead(nn.Module):
    """
    A quantum circuit that receives a scalar activation and returns the
    expectation of Pauli‑Z.  The circuit is implemented with
    torchquantum and executes on a CPU backend.  It is a direct
    counterpart to the classical `HybridHead` above.
    """

    def __init__(
        self,
        n_qubits: int,
        shift: float = np.pi / 2,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shift = shift
        self.device = device
        self._build_circuit()

    def _build_circuit(self):
        import torchquantum as tq

        self.qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=1, device=self.device)
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "h", "wires": [0]},
                {"input_idx": [0], "func": "ry", "wires": [0]},
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward a scalar or 1‑D tensor through the circuit and return the
        expectation value of Pauli‑Z.
        """
        thetas = inputs.detach().cpu().numpy()
        if thetas.ndim == 0:
            thetas = thetas.reshape(1)
        outputs = []
        for theta in thetas:
            self.qdev.reset()
            self.encoder(self.qdev, torch.tensor([theta], device=self.device))
            meas = self.measure(self.qdev)
            outputs.append(meas[0].item())
        return torch.tensor(outputs, dtype=torch.float32, device=self.device)


# --------------------------------------------------------------------------- #
# Unified quantum estimator
# --------------------------------------------------------------------------- #
class FastHybridEstimator:
    """
    Evaluates a quantum model or a hybrid classical‑quantum model.

    Parameters
    ----------
    model : nn.Module
        Either a pure quantum module (e.g. ``QModel``) or a hybrid
        model that ends with a ``HybridQuantumHead``.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable
            Callables that map a model output tensor to a scalar or a list
            of scalars.  For pure quantum models the output is a tensor
            of expectation values.
        parameter_sets : sequence
            Sequence of parameter vectors to evaluate.
        seed : int, optional
            Random seed for reproducibility (used only if the model
            internally adds noise).

        Returns
        -------
        List[List[float]]
            Nested list where each inner list contains the observable
            values for a single parameter set.
        """
        observables = list(observables) or [lambda o: o.mean(dim=-1)]

        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
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


__all__ = ["FastHybridEstimator", "HybridQuantumHead"]
