"""Hybrid quantum convolutional filter that encodes image patches into a
parameterized RX circuit and measures the average |1> probability.

The interface mirrors the classical HybridConvFilter so that the
quantum filter can be swapped in place of the classical one without
changing downstream code.  The filter operates on 2×2 patches with
stride 2, matching the dimensionality of the classical counterpart.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import torch


class HybridConvFilter:
    """Quantum filter that processes 2×2 image patches.

    The filter constructs a parameterized circuit consisting of
    an RX gate per qubit followed by a short random layer.  The
    circuit is executed on a simulator, and the average probability
    of measuring |1> over all qubits is returned as the feature
    value for the patch.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 2,
        backend=None,
        shots: int = 1024,
        threshold: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.seed = seed
        self._build_circuit()

    def _build_circuit(self) -> None:
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
        ]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2, seed=self.seed)
        self.circuit.measure_all()

    def _run_single_patch(self, patch: torch.Tensor) -> float:
        """Execute the circuit for a single 2×2 patch."""
        # Convert patch to a flat numpy array
        patch_flat = np.reshape(patch.detach().cpu().numpy(), (self.n_qubits,))
        # Bind parameters based on a threshold
        param_bind = {
            theta: np.pi if val > self.threshold else 0
            for theta, val in zip(self.theta, patch_flat)
        }
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result().get_counts(self.circuit)
        # Compute average probability of measuring |1> across all qubits
        counts = 0
        for bitstring, freq in result.items():
            ones = sum(int(bit) for bit in bitstring)
            counts += ones * freq
        return counts / (self.shots * self.n_qubits)

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the quantum filter to all patches of the batch.

        Parameters
        ----------
        data : torch.Tensor
            Input image of shape ``(B, 1, H, W)``.
        Returns
        -------
        torch.Tensor
            Feature map of shape ``(B, 1, H', W')`` where
            ``H' = (H - kernel_size) // stride + 1``.
        """
        B, C, H, W = data.shape
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        feature_map = torch.zeros(B, 1, H_out, W_out, device=data.device)
        for b in range(B):
            for i in range(H_out):
                for j in range(W_out):
                    patch = data[
                        b,
                        0,
                        i * self.stride : i * self.stride + self.kernel_size,
                        j * self.stride : j * self.stride + self.kernel_size,
                    ]
                    val = self._run_single_patch(patch)
                    feature_map[b, 0, i, j] = val
        return feature_map


class HybridConvClassifier:
    """Hybrid classifier that stacks the quantum filter and a linear head."""

    def __init__(
        self,
        num_classes: int = 10,
        kernel_size: int = 2,
        stride: int = 2,
        backend=None,
        shots: int = 1024,
        threshold: float = 0.5,
    ) -> None:
        self.qfilter = HybridConvFilter(
            kernel_size=kernel_size,
            stride=stride,
            backend=backend,
            shots=shots,
            threshold=threshold,
        )
        # Compute feature map size after convolution
        dummy = torch.zeros(1, 1, 28, 28)
        fmaps = self.qfilter.run(dummy)
        feat_dim = fmaps.view(1, -1).size(1)
        self.linear = torch.nn.Linear(feat_dim, num_classes)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning log‑softmax logits."""
        features = self.qfilter.run(x)
        logits = self.linear(features.view(features.size(0), -1))
        return torch.nn.functional.log_softmax(logits, dim=-1)


def Conv() -> HybridConvFilter:
    """Return a ready‑to‑use quantum filter instance."""
    return HybridConvFilter()


__all__ = ["HybridConvFilter", "HybridConvClassifier", "Conv"]
