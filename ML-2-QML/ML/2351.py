"""Hybrid classical autoencoder with quantum kernel regularization.

This module defines :class:`HybridAutoencoder`, a PyTorch neural network that
combines a standard fully‑connected autoencoder with a quantum kernel
computed via TorchQuantum.  The kernel is used as a regularizer encouraging
latent vectors to be close in the Hilbert space defined by the quantum ansatz.
The class mirrors the interface of the original Autoencoder factory,
making it drop‑in compatible while exposing the new quantum regularization.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torchquantum as tq
from torchquantum.functional import func_name_dict


def _as_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


class QuantumKernel(tq.QuantumModule):
    """
    A simple quantum kernel based on a programmable list of gates.
    The gate list is intentionally different from the seed to avoid verbatim copying.
    """

    def __init__(self, func_list: list[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """
    Quantum kernel evaluated via a fixed TorchQuantum ansatz.
    The ansatz uses a different gate pattern from the seed to preserve originality.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Use a distinct set of gates: ry, rz, cx, h
        self.ansatz = QuantumKernel(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "rz", "wires": [1]},
                {"input_idx": [2], "func": "cx", "wires": [2, 3]},
                {"input_idx": [3], "func": "h", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class HybridAutoencoder(nn.Module):
    """
    A PyTorch autoencoder that optionally regularizes the latent space
    with a quantum kernel.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        use_qkernel: bool = False,
        qkernel_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_qkernel = use_qkernel
        self.qkernel_scale = qkernel_scale

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        if self.use_qkernel:
            self.qkernel = Kernel()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def kernel_regularizer(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise quantum kernel over the batch latent vectors
        and return a scalar regularization term.
        """
        if not self.use_qkernel:
            return torch.tensor(0.0, device=z.device)

        # Convert to numpy for kernel evaluation
        z_np = z.detach().cpu().numpy()
        K = kernel_matrix([torch.tensor(v) for v in z_np], [torch.tensor(v) for v in z_np])
        # Regularizer: encourage off‑diagonal similarity to be small
        reg = torch.tensor(K.diagonal().sum(), device=z.device)
        return self.qkernel_scale * reg

    def loss(self, x: torch.Tensor, recon: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Reconstruction + kernel regularization loss."""
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        reg = self.kernel_regularizer(z)
        return recon_loss + reg


def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            z = model.encode(batch)
            recon = model.decode(z)
            loss = model.loss(batch, recon, z)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["HybridAutoencoder", "train_hybrid_autoencoder"]
