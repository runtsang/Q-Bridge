"""Hybrid autoencoder combining classical MLP with quantum kernel regularization.

The class ``QuantumHybridAutoencoder`` extends ``torch.nn.Module`` and uses a
classical encoder/decoder pair together with a quantum kernel (TorchQuantum)
to regularise the latent space.  The encoder and decoder are identical to the
simple fully‑connected autoencoder from the seed, but the latent vectors are
compared with a quantum kernel matrix during training.  The kernel is
implemented as ``QuantumKernelMethod.Kernel``, which internally uses a
parameterised quantum circuit and a ``QuantumDevice`` from TorchQuantum.

The training routine ``train_hybrid_autoencoder`` demonstrates how to
interleave a classical optimiser (Adam) with a quantum‑kernel based
regularisation term.  A small ``FastEstimator`` wrapper is also provided
to evaluate the quantum kernel for a batch of latent vectors, mirroring
the structure of the QML ``FastBaseEstimator``.

The module is fully importable and can be dropped into any PyTorch
project that already contains the ``AutoencoderConfig`` and ``AutoencoderNet``
definitions from the original seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# 1.  Classical auto‑encoder configuration and network (seed‑derived)
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Sequence[int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder (seed implementation)."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

# --------------------------------------------------------------------------- #
# 2.  Quantum kernel (seed‑derived, but wrapped for PyTorch use)
# --------------------------------------------------------------------------- #

try:
    import torchquantum as tq
    from torchquantum.functional import func_name_dict
except Exception:  # pragma: no cover
    # Dummy kernel if TorchQuantum is unavailable
    class DummyKernel(nn.Module):
        def __init__(self, gamma: float = 1.0) -> None:
            super().__init__()
            self.gamma = gamma

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    Kernel = DummyKernel
else:
    class KernalAnsatz(tq.QuantumModule):
        """Encodes each feature into a parameterised Ry rotation."""

        def __init__(self, func_list: Sequence[dict]) -> None:
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
        """Quantum RBF‑like kernel implemented with a fixed circuit."""

        def __init__(self, n_wires: int = 4) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
            self.ansatz = KernalAnsatz(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            self.ansatz(self.q_device, x, y)
            return torch.abs(self.q_device.states.view(-1)[0])

# --------------------------------------------------------------------------- #
# 3.  FastEstimator wrapper (seed‑derived)
# --------------------------------------------------------------------------- #

class FastEstimator:
    """Evaluate a quantum kernel for a batch of latent vectors.

    The wrapper mimics the structure of the original FastBaseEstimator
    but operates on tensors and returns a NumPy array.
    """

    def __init__(self, kernel: nn.Module) -> None:
        self.kernel = kernel

    def evaluate(self, batch_latent: torch.Tensor) -> np.ndarray:
        """Return the Gram matrix of the batch using the quantum kernel."""
        n = batch_latent.shape[0]
        gram = torch.zeros((n, n), dtype=torch.float32, device=batch_latent.device)
        for i in range(n):
            for j in range(n):
                gram[i, j] = self.kernel(batch_latent[i].unsqueeze(0), batch_latent[j].unsqueeze(0))
        return gram.cpu().numpy()

# --------------------------------------------------------------------------- #
# 4.  Hybrid auto‑encoder with quantum‑kernel regularisation
# --------------------------------------------------------------------------- #

class QuantumHybridAutoencoder(nn.Module):
    """Hybrid auto‑encoder that augments classical reconstruction with a
    quantum‑kernel based regulariser.

    Parameters
    ----------
    config : AutoencoderConfig
        Classical architecture parameters.
    use_quantum_kernel : bool, optional
        Whether to include the quantum kernel regularisation term.
    kernel_gamma : float, optional
        Gamma parameter for the quantum kernel (used only when
        `use_quantum_kernel` is True).
    """

    def __init__(
        self,
        config: AutoencoderConfig,
        use_quantum_kernel: bool = True,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = AutoencoderNet(config)
        self.decoder = self.encoder.decoder
        self.use_qk = use_quantum_kernel
        if self.use_qk:
            self.kernel = Kernel()
            self.kernel_gamma = kernel_gamma
            self.estimator = FastEstimator(self.kernel)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))

    def reconstruction_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(recon, target, reduction="mean")

    def quantum_kernel_regulariser(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute a simple trace‑based regulariser from the kernel matrix."""
        gram = self.estimator.evaluate(latent)
        # Encourage orthogonality between latent vectors
        diag = np.diag(gram)
        off_diag = gram - np.diagflat(diag)
        return torch.tensor(off_diag.sum() / (gram.shape[0] * (gram.shape[0] - 1)), device=latent.device)

    def loss(self, recon: torch.Tensor, target: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        loss = self.reconstruction_loss(recon, target)
        if self.use_qk:
            loss += 0.01 * self.quantum_kernel_regulariser(latent)
        return loss

def train_hybrid_autoencoder(
    model: QuantumHybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> List[float]:
    """Train the hybrid model with a simple Adam optimiser.

    The routine mirrors the classical training loop from the seed but
    injects the quantum‑kernel regulariser into the loss.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            latent = model.encoder(batch)
            recon = model.decoder(latent)
            loss = model.loss(recon, batch, latent)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history

__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "Kernel",
    "FastEstimator",
    "QuantumHybridAutoencoder",
    "train_hybrid_autoencoder",
]
