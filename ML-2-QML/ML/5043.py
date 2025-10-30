"""Hybrid estimator combining classical neural networks, self‑attention,
and convolutional preprocessing.

The module retains the original lightweight FastBaseEstimator and
FastEstimator classes, but extends them with a UnifiedEstimator that
can evaluate pure classical models, pure quantum models, or hybrid
models that combine classical preprocessing with a quantum core.
"""

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable, Union

# ----------  Core lightweight estimator ----------
class FastBaseEstimator:
    """Evaluate a PyTorch model over a batch of input parameters."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._to_tensor(params)
                outputs = self.model(inputs)
                row = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

    @staticmethod
    def _to_tensor(values: Sequence[float]) -> torch.Tensor:
        t = torch.as_tensor(values, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t

class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot‑noise to deterministic outputs."""
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
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

# ---------- Classical preprocessing primitives ----------
class ClassicalSelfAttention:
    """Multi‑head self‑attention implemented with PyTorch tensors."""
    def __init__(self, embed_dim: int = 4) -> None:
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # reshape parameters into weight matrices
        q_w = torch.from_numpy(rotation_params.reshape(self.embed_dim, -1)).float()
        k_w = torch.from_numpy(entangle_params.reshape(self.embed_dim, -1)).float()
        x = torch.from_numpy(inputs).float()
        Q = x @ q_w.t()
        K = x @ k_w.t()
        V = x
        scores = torch.softmax(Q @ K.t() / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ V).numpy()

class ConvFilter(nn.Module):
    """2‑D convolution filter that mimics a quantum quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        t = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
        logits = self.conv(t)
        act = torch.sigmoid(logits - self.threshold)
        return act.mean().item()

# ---------- Hybrid estimator ----------
class HybridEstimator:
    """
    Unified estimator that can evaluate:
      * Purely classical PyTorch models.
      * Purely quantum models (implementations that expose a ``evaluate`` method).
      * Hybrid models that combine a classical pre‑processor with a quantum core.
    """

    def __init__(
        self,
        model: Union[nn.Module, Callable],
        *,
        preprocessor: Callable | None = None,
        noise_shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model
            Either a PyTorch ``nn.Module`` or an object that implements
            ``evaluate(observables, parameter_sets)``.
        preprocessor
            Optional callable that transforms raw parameters before they reach
            the model.  Typical choices are ``ClassicalSelfAttention`` or
            ``ConvFilter``.
        noise_shots
            If supplied, Gaussian noise with standard deviation ``1/shots`` is
            added to every output to emulate quantum measurement statistics.
        seed
            RNG seed for reproducible noise.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.noise_shots = noise_shots
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        # Apply optional classical pre‑processing
        if self.preprocessor is not None:
            parameter_sets = [self.preprocessor.run(np.array(params)) for params in parameter_sets]
        # Dispatch to underlying model
        if hasattr(self.model, "evaluate"):
            raw = self.model.evaluate(observables, parameter_sets)
        else:
            # Assume PyTorch model
            fast = FastEstimator(self.model) if self.noise_shots else FastBaseEstimator(self.model)
            raw = fast.evaluate(observables, parameter_sets)
        # Inject noise if required
        if self.noise_shots is not None:
            noisy = []
            for row in raw:
                noisy_row = [float(self._rng.normal(mean, max(1e-6, 1 / self.noise_shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy
        return raw

__all__ = ["FastBaseEstimator", "FastEstimator", "ClassicalSelfAttention",
           "ConvFilter", "HybridEstimator"]
