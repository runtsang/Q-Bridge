"""Quantum kernel estimator using TorchQuantum with optional shot noise.

The quantum implementation mirrors the classical estimator but
computes the kernel as the overlap of two parametrically
prepared states.  The ansatz encodes the two feature vectors with
RY rotations, followed by an inverse sequence that cancels the
encoding and yields the fidelity.  Gaussian shot noise can be
superimposed to emulate a finite‑shot measurement, matching the
interface of the classical FastEstimator.

The design keeps the API identical to the classical variant, so
experiments can switch between the two seamlessly.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from collections.abc import Iterable, Sequence
from typing import List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class QuantumKernelAnsatz(tq.QuantumModule):
    """Encodes two feature vectors x and y into a quantum state."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Build a list of instructions that will be applied during forward.
        self.ansatz = [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(self.n_wires)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Apply the encoding and the inverse encoding."""
        q_device.reset_states(x.shape[0])
        for info in self.ansatz:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Apply the inverse encoding using -y.
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class FastBaseEstimator:
    """Base class that evaluates a callable quantum kernel for batches of inputs."""
    def __init__(self, model: tq.QuantumModule) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs]
        results: List[List[float]] = []
        for params in parameter_sets:
            # Split the concatenated parameters back into x and y.
            split = len(params) // 2
            x = torch.tensor(params[:split], dtype=torch.float32).reshape(1, -1)
            y = torch.tensor(params[split:], dtype=torch.float32).reshape(1, -1)
            outputs = self.model(x, y)
            row: List[float] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = float(value.mean().cpu())
                else:
                    scalar = float(value)
                row.append(scalar)
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to a deterministic quantum kernel."""
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
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


class QuantumKernelEstimator(tq.QuantumModule):
    """Quantum kernel estimator that mimics the classical API.

    Parameters
    ----------
    n_wires : int, optional
        Number of qubits used in the ansatz.
    shots : int | None, optional
        If provided, Gaussian noise with std 1/√shots is added to each
        kernel evaluation to emulate finite‑shot measurements.
    seed : int | None, optional
        Random seed for reproducibility of the noise.
    """
    def __init__(self, n_wires: int = 4, shots: Optional[int] = None, seed: Optional[int] = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernelAnsatz(self.n_wires)
        self.shots = shots
        self.seed = seed

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for a single pair (x, y)."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Return the Gram matrix between `a` and `b`.

        The method uses the FastEstimator pattern to optionally add
        Gaussian noise.
        """
        shots = shots if shots is not None else self.shots
        seed = seed if seed is not None else self.seed

        param_sets: List[Sequence[float]] = []
        for x in a:
            for y in b:
                param_sets.append(torch.cat([x, y]).tolist())

        estimator = FastEstimator(self, shots=shots, seed=seed)
        results = estimator.evaluate([lambda out: out], param_sets)
        matrix = np.array([row[0] for row in results]).reshape(len(a), len(b))
        return matrix


__all__ = ["QuantumKernelAnsatz", "FastBaseEstimator", "FastEstimator", "QuantumKernelEstimator"]
