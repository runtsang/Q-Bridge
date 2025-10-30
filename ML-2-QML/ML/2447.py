import torch
from torch import nn
import numpy as np

class SharedClassName(nn.Module):
    """Hybrid feed‑forward regressor that prepends a classical self‑attention
    block before the original EstimatorQNN architecture.  The attention
    block learns a context‑aware representation of the input features,
    allowing the downstream linear layers to operate on a richer
    embedding.  This design mirrors the quantum self‑attention circuit
    in the QML counterpart, providing a direct classical analogue
    for comparative studies."""
    def __init__(self, input_dim: int, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Parameters for rotating and entangling the input in the attention block
        self.rotation_params = nn.Parameter(torch.randn(input_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(input_dim, embed_dim))
        # Core regression network (original EstimatorQNN)
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def _classical_attention(self, rotation_params: np.ndarray,
                            entangle_params: np.ndarray,
                            inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                                dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                              dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for the attention routine
        inp_np = inputs.detach().cpu().numpy()
        attn_out = self._classical_attention(
            self.rotation_params.detach().cpu().numpy(),
            self.entangle_params.detach().cpu().numpy(),
            inp_np
        )
        # Feed the attended representation to the regression network
        return self.regressor(torch.from_numpy(attn_out).to(inputs.device))

__all__ = ["SharedClassName"]
