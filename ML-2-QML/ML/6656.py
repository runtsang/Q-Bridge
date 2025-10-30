import torch
from torch import nn
import torch.nn.functional as F

class EstimatorQNN(nn.Module):
    """
    A richer feed‑forward regressor that keeps the original two‑input
    interface but adds:
      * A depth‑wise residual block that can be optionally turned on.
      * Two‑stage feature extraction (first 8‑unit layer, then 4‑unit layer)
        with a learnable activation choice.
      * A hybrid loss that accepts a classical loss and a quantum penalty
        term, enabling joint training with a QNN.
    """

    def __init__(self,
                 hidden_units: int = 8,
                 residual: bool = False,
                 act: str = "tanh"):
        super().__init__()
        if act not in {"tanh", "relu", "gelu"}:
            raise ValueError(f"Unsupported activation: {act}")
        act_fn = {"tanh": nn.Tanh(),
                  "relu": nn.ReLU(),
                  "gelu": nn.GELU()}[act]
        self.activation = act_fn

        # Feature extraction layers
        self.feature = nn.Sequential(
            nn.Linear(2, hidden_units),
            self.activation,
            nn.Linear(hidden_units, 4),
            self.activation,
        )

        # Optional residual connection
        self.residual_layer = nn.Linear(2, 4) if residual else None

        # Output layer
        self.output = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 2).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1).
        """
        h = self.feature(x)
        if self.residual_layer is not None:
            h = h + self.residual_layer(x)
        return self.output(h)

    def hybrid_loss(self,
                    preds: torch.Tensor,
                    target: torch.Tensor,
                    quantum_penalty: float = 0.0) -> torch.Tensor:
        """
        Compute a hybrid loss combining MSE with an optional quantum penalty.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values.
        target : torch.Tensor
            Ground truth values.
        quantum_penalty : float, optional
            Weight of the quantum penalty term.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        mse = F.mse_loss(preds, target)
        if quantum_penalty!= 0.0:
            penalty = quantum_penalty * preds.mean().abs()
            return mse + penalty
        return mse

__all__ = ["EstimatorQNN"]
