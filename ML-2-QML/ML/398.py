import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen(nn.Module):
    """
    Extended classical sampler network.

    Features:
      * Two hidden layers with batch‑normalisation and dropout for better regularisation.
      * ReLU activations to avoid vanishing gradients.
      * Log‑softmax output for numerical stability.
      * Utility `sample` method to draw discrete samples from the learned distribution.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int, int] = (8, 4), output_dim: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log‑probabilities over the output classes."""
        logits = self.net(x)
        return F.log_softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1, temperature: float = 1.0) -> torch.Tensor:
        """
        Draw discrete samples from the categorical distribution defined by the network output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).
        num_samples : int
            Number of samples to draw per input.
        temperature : float
            Soft‑max temperature for Gumbel‑softmax trick.
        """
        log_probs = self.forward(x)
        probs = torch.exp(log_probs)
        # Gumbel‑softmax reparameterisation
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + 1e-10) + 1e-10)
        y = F.softmax((log_probs + gumbel_noise) / temperature, dim=-1)
        samples = torch.multinomial(y, num_samples=num_samples, replacement=True)
        return samples
