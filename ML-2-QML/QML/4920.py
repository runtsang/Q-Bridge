"""Quantum-enhanced quanvolution model combining a quantum kernel, classical auto‑encoder and a variational sampler.

The class `QuanvolutionHybrid` mirrors the classical `QuanvolutionHybrid` but replaces the
convolutional filter with a 4‑qubit quantum kernel that processes each 2×2 image
patch.  The concatenated kernel output is compressed by a lightweight
classical auto‑encoder, and the resulting latent vector is fed into a
Qiskit `SamplerQNN` that implements a 10‑qubit variational circuit for
classification.  The implementation is fully importable and can be dropped
into either a pure‑quantum or a hybrid training pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from typing import Tuple

# ---------- Quantum kernel for 2×2 patches ----------
class PatchKernel(tq.QuantumModule):
    """
    Encodes a 2×2 image patch (4 pixel values) into a 4‑dimensional
    quantum feature vector by applying Ry gates to 4 qubits and measuring
    the expectation of Pauli‑Z on each qubit.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 4) – the 4 pixel values of a patch.

        Returns
        -------
        torch.Tensor
            Shape (batch, 4) – expectation values of Pauli‑Z on each qubit.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=device)
        for i in range(self.n_wires):
            qdev.ry(x[:, i], wires=i)
        exp_vals = torch.stack(
            [qdev.get_expectation(tq.PauliZ, wires=[i]) for i in range(self.n_wires)],
            dim=1,
        )
        return exp_vals

# ---------- Classical auto‑encoder ----------
class AutoencoderNet(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        encoder = []
        in_dim = 4 * 14 * 14  # flatten size after kernel
        for hidden in hidden_dims:
            encoder.append(nn.Linear(in_dim, hidden))
            encoder.append(nn.ReLU())
            if dropout > 0.0:
                encoder.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder.append(nn.Linear(in_dim, hidden))
            decoder.append(nn.ReLU())
            if dropout > 0.0:
                decoder.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder.append(nn.Linear(in_dim, 4 * 14 * 14))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

# ---------- Sampler classifier ----------
class SamplerClassifier:
    """
    Wraps a Qiskit SamplerQNN that implements a 10‑qubit variational circuit.
    The latent vector of length 10 is interpreted as input parameters for the
    Ry gates on each qubit; a simple entangling layer follows, and the
    measurement outcomes are interpreted as class probabilities.
    """
    def __init__(self, num_qubits: int = 10) -> None:
        self.num_qubits = num_qubits
        # Build a parameterised circuit
        self.input_params = ParameterVector('input', num_qubits)
        self.weight_params = ParameterVector('weight', num_qubits)  # one weight per qubit
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.ry(self.input_params[i], i)
        # Simple entangling block
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        for i in range(num_qubits - 1):
            qc.cx(i + 1, i)
        # Additional Ry gates as variational weights
        for i in range(num_qubits):
            qc.ry(self.weight_params[i], i)
        qc.measure_all()

        self.sampler = StatevectorSampler()
        self.classifier = SamplerQNN(
            circuit=qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            interpret=lambda x: x,
            output_shape=num_qubits,
            sampler=self.sampler,
        )

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent : torch.Tensor
            Shape (batch, num_qubits) – the latent vector from the auto‑encoder.

        Returns
        -------
        torch.Tensor
            Shape (batch, num_qubits) – log‑softmax of class probabilities.
        """
        probs = []
        for i in range(latent.shape[0]):
            latent_np = latent[i].cpu().numpy().reshape(1, -1)
            result = self.classifier(latent_np)
            probs.append(result['probs'][0])  # shape (num_qubits,)
        probs = torch.tensor(probs, dtype=latent.dtype, device=latent.device)
        log_probs = torch.log(probs + 1e-10)
        return log_probs

# ---------- Hybrid Model ----------
class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum‑enhanced quanvolution model.  The architecture mirrors the
    classical `QuanvolutionHybrid` but replaces the convolutional filter
    with a 4‑qubit quantum kernel that operates on each 2×2 patch.  The
    concatenated kernel output is compressed by a classical auto‑encoder
    and the resulting latent vector is fed into a variational sampler
    network for classification.
    """
    def __init__(self, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_kernel = PatchKernel()
        self.autoencoder = AutoencoderNet(latent_dim=latent_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.classifier = SamplerClassifier(num_qubits=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 1, 28, 28) – input images.

        Returns
        -------
        torch.Tensor
            Shape (batch, 10) – log‑softmax class probabilities.
        """
        bsz = x.shape[0]
        device = x.device
        # Extract 2×2 patches
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # shape (batch, 1, 14, 14, 2, 2)
        patches = patches.reshape(bsz, 14 * 14, 4)  # (batch, 196, 4)
        # Apply quantum kernel to each patch
        kernel_features = []
        for i in range(patches.shape[1]):
            patch = patches[:, i, :]  # (batch, 4)
            feat = self.patch_kernel(patch)  # (batch, 4)
            kernel_features.append(feat)
        kernel_features = torch.cat(kernel_features, dim=1)  # (batch, 784)
        # Compress with classical auto‑encoder
        latent = self.autoencoder.encode(kernel_features)  # (batch, latent_dim)
        # Classify with sampler
        log_probs = self.classifier(latent)  # (batch, 10)
        return log_probs

__all__ = ["QuanvolutionHybrid"]
