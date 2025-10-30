import torch
import torch.nn as nn
import torch.nn.functional as F


class FraudDetectionHybrid(nn.Module):
    """
    Classical LSTM + MLP classifier for fraud detection.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each transaction feature vector.
    lstm_hidden : int
        Size of the hidden state in the LSTM.
    lstm_layers : int
        Number of stacked LSTM layers.
    mlp_hidden : int
        Hidden dimensionality of the downstream MLP.
    """

    def __init__(
        self,
        input_dim: int = 2,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        mlp_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Fraud probability of shape (batch,).
        """
        lstm_out, _ = self.lstm(x)
        # use the last hidden state
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden).squeeze(-1)


__all__ = ["FraudDetectionHybrid"]
