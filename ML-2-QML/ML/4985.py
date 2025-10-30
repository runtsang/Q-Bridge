import torch
from torch import nn
from typing import Sequence

# Local helpers (imported from the same package)
from.Autoencoder import Autoencoder
from.SelfAttention import SelfAttention
from.FastEstimator import FastEstimator

class EstimatorQNN(nn.Module):
    """
    Hybrid regressor that integrates:
      1. A lightweight auto‑encoder (AutoencoderNet) that compresses the input.
      2. A classical self‑attention block that re‑weights the latent vector.
      3. A linear head that maps the attended vector to a scalar output.

    The design mirrors the quantum version but remains fully classical,
    enabling quick prototyping or CPU‑only training.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Encoder
        self.encoder = Autoencoder(
            input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        # Classical attention
        self.attention = SelfAttention()
        # Output head
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Encode
        latent = self.encoder.encode(inputs)
        # Prepare parameters for the attention block
        rot = latent.mean(dim=0, keepdim=True).to(torch.float32).cpu().numpy()
        ent = latent.std(dim=0, keepdim=True).to(torch.float32).cpu().numpy()
        # Apply attention (returns numpy array)
        attended = self.attention.run(rot_params=rot, entangle_params=ent, inputs=latent.cpu().numpy())
        attended_tensor = torch.as_tensor(attended, dtype=latent.dtype, device=latent.device)
        # Final regression
        return self.head(attended_tensor)

class FastEstimatorWrapper(FastEstimator):
    """
    Wraps EstimatorQNN to provide a FastEstimator compatible API.
    The ``predict`` method evaluates the model on a batch of inputs
    and optionally adds Gaussian shot noise.
    """
    def __init__(self, model: EstimatorQNN) -> None:
        super().__init__(model)

    def predict(
        self,
        inputs: torch.Tensor,
        shots: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        observables = [lambda out: out]
        param_sets = inputs.tolist()
        results = self.evaluate(observables, param_sets, shots=shots, seed=seed)
        return torch.tensor(results, dtype=torch.float32)

__all__ = ["EstimatorQNN", "FastEstimatorWrapper"]
