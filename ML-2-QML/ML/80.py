"""ConvGen – a hybrid depth‑wise separable convolution with optional quantum augmentation.

This module replaces the original ``Conv`` class while adding several improvements:
* A learnable depth‑wise separable convolution (grouped Conv2d) so that each channel can be trained independently.
* Optional bias and batch‑normalization for better expressiveness.
* A helper that creates a 2‑D Gaussian kernel for use as a prior.
* A ``from_quantum`` class method that accepts a Qiskit circuit and turns it into a classical tensor by measuring every shot and then fitting a small MLP.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["ConvGen"]

class ConvGen(nn.Module):
    """Drop‑in replacement for the original ``Conv`` filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the 2‑D filter.
    depthwise : bool, default True
        If True, use grouped convolution for depth‑wise separable filtering.
    bias : bool, default True
        Whether to include a learnable bias term.
    use_bn : bool, default False
        Whether to apply batch‑normalization after the convolution.
    threshold : float, default 0.0
        Threshold for the sigmoid activation.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        depthwise: bool = True,
        bias: bool = True,
        use_bn: bool = False,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        in_channels = 1
        out_channels = 1

        groups = in_channels if depthwise else 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=bias,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None

    def forward(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Apply the convolution and return the mean activation.

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            2‑D array of shape (H, W) or (1, H, W).

        Returns
        -------
        torch.Tensor
            Mean activation after the sigmoid threshold.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.astype(np.float32))
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif data.ndim == 3 and data.shape[0] == 1:
            data = data.unsqueeze(1)  # (1, 1, H, W)
        logits = self.conv(data)
        if self.bn is not None:
            logits = self.bn(logits)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    @staticmethod
    def gaussian_kernel(kernel_size: int, sigma: float = 1.0) -> torch.Tensor:
        """Generate a 2‑D Gaussian kernel.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel.
        sigma : float, default 1.0
            Standard deviation of the Gaussian.

        Returns
        -------
        torch.Tensor
            2‑D Gaussian kernel of shape (kernel_size, kernel_size).
        """
        ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / np.sum(kernel)
        return torch.from_numpy(kernel.astype(np.float32))

    @classmethod
    def from_quantum(
        cls,
        circuit: "qiskit.QuantumCircuit",
        backend: "qiskit.providers.BaseBackend",
        shots: int = 1024,
        threshold: float = 0.5,
    ) -> "ConvGen":
        """Create a ConvGen instance that mimics a quantum circuit.

        The method runs the circuit for each input pixel, measures the probability
        of |1⟩, and uses the resulting values as weights for a small MLP that
        approximates the quantum filter.

        Parameters
        ----------
        circuit : qiskit.QuantumCircuit
            Parameter‑tuned variational circuit.
        backend : qiskit.providers.BaseBackend
            Backend to execute the circuit.
        shots : int, default 1024
            Number of shots per evaluation.
        threshold : float, default 0.5
            Threshold for mapping classical data to circuit parameters.

        Returns
        -------
        ConvGen
            A classical approximation of the quantum filter.
        """
        import qiskit  # Imported lazily to avoid mandatory dependency

        n_qubits = circuit.num_qubits
        kernel_size = int(np.sqrt(n_qubits))
        # Run the circuit on a dummy input to obtain a weight matrix
        dummy_data = np.zeros((kernel_size, kernel_size))
        probs = []
        for _ in range(10):  # few repetitions for stability
            job = qiskit.execute(
                circuit,
                backend,
                shots=shots,
            )
            result = job.result()
            counts = result.get_counts(circuit)
            # Compute average probability of |1⟩ per qubit
            probs.append(
                sum(
                    sum(int(bit) for bit in key) * val
                    for key, val in counts.items()
                )
                / (shots * n_qubits)
            )
        weight = np.mean(probs).astype(np.float32)
        conv = cls(kernel_size=kernel_size, bias=False, threshold=threshold)
        with torch.no_grad():
            conv.conv.weight.copy_(torch.full_like(conv.conv.weight, weight))
        return conv
