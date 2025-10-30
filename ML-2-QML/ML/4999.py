import torch
from torch import nn
import torch.nn.functional as F

class FraudDetectionHybrid(nn.Module):
    """Classical fraud‑detection architecture.
    Combines a light feed‑forward core (inspired by the photonic analogues)
    with optional sequence modelling (LSTM) and a small regression head
    (inspired by the EstimatorQNN example).  The design mirrors the
    original `FraudDetection` seed while adding the LSTM and regressor
    blocks to explore time‑series behaviour and risk‑scoring.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        use_lstm: bool = False,
        lstm_hidden: int = 16,
        lstm_layers: int = 1,
        regression: bool = True,
    ) -> None:
        super().__init__()
        self.use_lstm = use_lstm
        self.regression = regression

        # Core linear stack
        self.core = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Optional LSTM for sequential fraud patterns
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
            )
            lstm_output_dim = lstm_hidden
        else:
            lstm_output_dim = hidden_dim

        # Final classification head
        self.classifier = nn.Linear(lstm_output_dim, 1)

        # Small regression head (EstimatorQNN style)
        if regression:
            self.regressor = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )
        else:
            self.regressor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, features) for sequential data or
            (batch, features) for static data.

        Returns
        -------
        torch.Tensor
            Classification log‑probabilities if `regression` is False,
            otherwise a tuple `(logits, risk_score)`.
        """
        if x.dim() == 2:
            # Static case
            out = self.core(x)
            logits = self.classifier(out)
            if self.regression:
                risk = self.regressor(x)
                return logits, risk
            return logits
        elif x.dim() == 3:
            # Sequential case
            out = self.core(x)  # (batch, seq, hidden)
            out, _ = self.lstm(out)
            logits = self.classifier(out)
            if self.regression:
                risk = self.regressor(x[:, -1, :])  # last timestep
                return logits, risk
            return logits
        else:
            raise ValueError("Input tensor must be 2 or 3 dimensional")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of fraud."""
        logits, *_ = self.forward(x)
        return torch.sigmoid(logits)

__all__ = ["FraudDetectionHybrid"]
