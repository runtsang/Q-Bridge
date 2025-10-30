import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, Optional

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class HybridConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    n_qubits: int = 1
    shots: int = 1024

class HybridFCL_AE(nn.Module):
    """
    Hybrid quantum-classical autoencoder.
    The quantum layer acts as an encoder producing a latent vector
    of dimension `latent_dim`. A classical decoder reconstructs the
    input from this latent representation.
    """
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.qc_run: Optional[Callable[[Iterable[float]], torch.Tensor]] = None

        # Classical decoder: maps latent_dim -> input_dim
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in config.hidden_dims[::-1]:
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: quantum encoder followed by classical decoder.
        Expects `x` to be a 1‑D tensor of length `latent_dim`.
        """
        if self.qc_run is None:
            raise RuntimeError("Quantum circuit not attached. Call `set_quantum_circuit`.")
        # Quantum encoder returns a numpy array of latent values
        latent_np = self.qc_run(x.detach().cpu().numpy().tolist())
        latent = torch.as_tensor(latent_np, dtype=torch.float32, device=x.device)
        return self.decoder(latent)

    def set_quantum_circuit(self, qc: Callable[[Iterable[float]], torch.Tensor]) -> None:
        """Attach a callable that implements the quantum encoder."""
        self.qc_run = qc

def HybridFCL_AE_factory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    n_qubits: int = 1,
    shots: int = 1024,
) -> HybridFCL_AE:
    config = HybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        n_qubits=n_qubits,
        shots=shots,
    )
    return HybridFCL_AE(config)

__all__ = ["HybridFCL_AE", "HybridFCL_AE_factory", "HybridConfig"]
