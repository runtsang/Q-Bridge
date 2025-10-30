import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence, List

class HybridFCL(nn.Module):
    """
    Classical fully‑connected layer that mimics the behaviour of the
    original FCL example.  It offers the same ``run`` and ``evaluate``
    interface as the quantum counterpart, enabling direct comparison
    and batched evaluation with optional Gaussian shot noise.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the layer for a single set of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Sequence of parameters to feed into the linear layer.

        Returns
        -------
        np.ndarray
            The mean tanh activation over the linear output.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

    def evaluate(
        self,
        thetas_list: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Batch evaluate many parameter sets with optional shot noise.

        Parameters
        ----------
        thetas_list : Sequence[Sequence[float]]
            List of parameter sequences to evaluate.
        shots : int, optional
            If provided, Gaussian shot noise with variance 1/shots is added.
        seed : int, optional
            Seed for the random number generator.

        Returns
        -------
        List[List[float]]
            Nested list where each inner list contains the expectation
            value for the corresponding parameter set.
        """
        results: List[List[float]] = []
        if shots is None:
            for params in thetas_list:
                results.append([float(self.run(params))])
            return results

        rng = np.random.default_rng(seed)
        for params in thetas_list:
            mean = float(self.run(params))
            noisy = rng.normal(mean, max(1e-6, 1 / shots))
            results.append([float(noisy)])
        return results
