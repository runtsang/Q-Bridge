import torch
from torch import nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastHybridEstimator:
    """Hybrid estimator that can wrap a pure PyTorch model or a hybrid model that includes a quantum kernel.

    The estimator exposes the same evaluation interface as the original FastBaseEstimator but adds:
      * support for a noise model (Gaussian shot noise)
      * convenience factory methods that mirror the quantum build functions
      * optional device selection for GPU acceleration
    """

    def __init__(self, model: nn.Module | Callable[[torch.Tensor], torch.Tensor], device: str | torch.device | None = None) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables:
            Sequence of callables that map the model output tensor to a scalar.
            If None a single observable returning the mean of all output elements is used.
        parameter_sets:
            Sequence of float sequences.  Each inner sequence is interpreted as a batch of
            parameters fed to the model.  For a pure neural network these are the input
            features; for a hybrid model they can be the encoded parameters.
        shots:
            Optional shot noise model.  If provided, Gaussian noise with std = 1/√shots
            is added to each returned mean.
        seed:
            Random seed for reproducibility of the shot noise.

        Returns
        -------
        list[list[float]]:
            Nested list where each inner list corresponds to a parameter set.
        """
        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]

        if parameter_sets is None:
            parameter_sets = []

        results: List[List[float]] = []
        rng = np.random.default_rng(seed)

        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
            if shots is not None:
                row = [rng.normal(mean, max(1e-6, 1 / shots)) for mean in row]
            results.append(row)
        return results

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int = 1) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Construct a simple feed‑forward classifier and return metadata identical to the quantum interface.

        Returns
        -------
        network:
            nn.Sequential model consisting of ``depth`` hidden layers followed by a 2‑class head.
        encoding:
            list of input feature indices (identity mapping).
        weight_sizes:
            list of the number of trainable parameters per layer.
        observables:
            placeholder list of observable indices; used only to keep API parity.
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

    @staticmethod
    def build_quanvolution_filter(kernel_size: int = 2, num_filters: int = 4) -> nn.Module:
        """Return a classical convolutional layer that mimics the quantum quanvolution filter.

        Parameters
        ----------
        kernel_size:
            Size of the square patch.  The implementation uses a 2D convolution with
            ``num_filters`` output channels to emulate a random quantum kernel.
        num_filters:
            Number of output channels; each channel can be interpreted as a different
            quantum feature map.
        """
        class QuanvolutionFilter(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, num_filters, kernel_size=kernel_size, stride=kernel_size)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                # Shape: (batch, 1, H, W) -> (batch, num_filters, H//k, W//k)
                patch = self.conv(x)
                return patch.view(x.size(0), -1)

        return QuanvolutionFilter()

__all__ = ["FastHybridEstimator"]
