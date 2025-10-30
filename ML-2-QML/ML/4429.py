import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

from.Conv import Conv
from.Autoencoder import Autoencoder, AutoencoderNet
from.EstimatorQNN import EstimatorQNN

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridEstimator:
    """Hybrid classical estimator that evaluates a PyTorch model with optional preprocessing and noise.

    Features:
    * Accepts any nn.Module, or factory functions such as EstimatorQNN, Conv, Autoencoder.
    * Optional convolutional filter applied to each input before the model.
    * Optional autoencoder encoding of inputs.
    * Optional Gaussian shot noise added to deterministic outputs.
    * Supports a list of scalar observables; defaults to mean over the last dimension.
    """

    def __init__(
        self,
        model: Union[nn.Module, Callable[[], nn.Module]],
        *,
        conv_filter: Optional[Callable[[], nn.Module]] = None,
        autoencoder: Optional[Union[AutoencoderNet, Callable[[], AutoencoderNet]]] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if isinstance(model, nn.Module):
            self.model = model
        else:
            self.model = model()
        self.conv_filter = conv_filter() if conv_filter is not None else None
        self.autoencoder = autoencoder() if autoencoder is not None else None
        self.shots = shots
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _preprocess(self, params: torch.Tensor) -> torch.Tensor:
        x = params
        if self.conv_filter is not None:
            k = self.conv_filter.kernel_size
            x = x.view(-1, 1, k, k)
            x = self.conv_filter(x)
        if self.autoencoder is not None:
            x = self.autoencoder.encode(x)
        return x

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                processed = self._preprocess(inputs)
                outputs = self.model(processed)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if self.shots is not None:
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy
        return results

__all__ = ["HybridEstimator"]
