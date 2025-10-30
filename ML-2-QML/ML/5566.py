"""Hybrid classical–quantum kernel module combining RBF, TorchQuantum, fraud‑layer, and convolutional preprocessing.

The module exposes three main classes:
* ClassicalKernel – standard RBF kernel.
* QuantumKernel – TorchQuantum variational kernel.
* HybridKernel – selects between the two, optionally applies a fraud‑layer transformation
  and a 2×2 convolutional filter, and offers a noisy estimator interface.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Callable

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Classical RBF kernel – kept for compatibility
# --------------------------------------------------------------------------- #
class ClassicalKernalAnsatz(nn.Module):
    """RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class ClassicalKernel(nn.Module):
    """Simple wrapper around :class:`ClassicalKernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = ClassicalKernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

# --------------------------------------------------------------------------- #
# Quantum kernel – TorchQuantum implementation
# --------------------------------------------------------------------------- #
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumKernalAnsatz(tq.QuantumModule):
    """Encodes data through a programmable list of gates."""
    def __init__(self, func_list: List[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states."""
    def __init__(self, n_wires: int = 4, func_list: List[dict] | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        if func_list is None:
            func_list = [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        self.ansatz = QuantumKernalAnsatz(func_list)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

# --------------------------------------------------------------------------- #
# Hybrid kernel – combines classical, quantum, fraud‑layer and convolution
# --------------------------------------------------------------------------- #
# Import utilities from the other seed modules
try:
    from.FastBaseEstimator import FastBaseEstimator, FastEstimator
except Exception:
    FastBaseEstimator = FastEstimator = None

try:
    from.FraudDetection import FraudLayerParameters
except Exception:
    FraudLayerParameters = None

try:
    from.Conv import Conv
except Exception:
    Conv = None

class HybridKernel(nn.Module):
    """
    Unified kernel that can operate in classical RBF mode, quantum mode, or a hybrid of both.
    Optional fraud‑layer transformation and 2×2 convolutional preprocessing are supported.
    The class also exposes a lightweight estimator that can add Gaussian shot noise.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        use_quantum: bool = False,
        quantum_func_list: List[dict] | None = None,
        fraud_params=None,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.kernel = QuantumKernel(4, quantum_func_list) if use_quantum else ClassicalKernel(gamma)

        # Fraud layer: transforms a 1‑D feature into a 2‑D vector before feeding into the kernel
        self.fraud_layer = None
        if fraud_params is not None:
            # Build a simple linear layer using the fraud params
            self.fraud_layer = self._build_fraud_layer(fraud_params, clip=True)

        # Convolutional filter
        self.conv_filter = Conv()
        self.conv_filter.kernel_size = conv_kernel_size
        self.conv_filter.threshold = conv_threshold

    # ----------------------------------------------------------------------- #
    # Internal helpers
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _build_fraud_layer(params, clip: bool) -> nn.Module:
        """
        Construct a 2‑input → 2‑output linear layer that mimics the photonic
        fraud‑detection layer.  The layer is used to map a single scalar feature
        into a 2‑D vector, which is then processed by the kernel.
        """
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi]],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        activation = nn.Tanh()
        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

        class FraudLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = linear
                self.activation = activation
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                outputs = self.activation(self.linear(inputs))
                outputs = outputs * self.scale + self.shift
                return outputs

        return FraudLayer()

    def _preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply the 2×2 convolution filter to each sample, duplicate the scalar
        into a 2‑D vector, pass it through the fraud layer (if any), and
        finally return a 1‑D vector ready for the kernel.
        """
        if data.ndim == 1:
            data = data.unsqueeze(0)
        N, D = data.shape
        # Assume each row is a flattened 2×2 patch
        patch = data.view(N, self.conv_filter.kernel_size, self.conv_filter.kernel_size)
        conv_out = torch.tensor([self.conv_filter.run(patch[i]) for i in range(N)])
        # Duplicate to 2‑D
        conv_out = conv_out.unsqueeze(-1).repeat(1, 2)
        if self.fraud_layer is not None:
            conv_out = self.fraud_layer(conv_out)
            # Collapse back to 1‑D by averaging
            conv_out = conv_out.mean(dim=-1, keepdim=True)
        return conv_out

    # ----------------------------------------------------------------------- #
    # Kernel interface
    # ----------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel value between two samples."""
        x_pre = self._preprocess(x)
        y_pre = self._preprocess(y)
        return self.kernel(x_pre, y_pre)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix between two sets of samples."""
        mat = np.array([[self.forward(a_i, b_j).item() for b_j in b] for a_i in a])
        return mat

    # ----------------------------------------------------------------------- #
    # Estimator interface
    # ----------------------------------------------------------------------- #
    def evaluate(
        self,
        model: nn.Module,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate a model for multiple parameter sets.  If ``shots`` is given,
        Gaussian shot noise is added to the deterministic outputs.

        The estimator works for both classical and quantum models.
        """
        # Choose the appropriate estimator
        if FastEstimator is not None and shots is not None:
            estimator = FastEstimator(model)
        else:
            estimator = FastBaseEstimator(model) if FastBaseEstimator is not None else model

        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = [
    "ClassicalKernalAnsatz",
    "ClassicalKernel",
    "QuantumKernalAnsatz",
    "QuantumKernel",
    "HybridKernel",
]
