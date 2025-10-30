import torch
from torch import nn
from typing import Tuple

class HybridEstimatorQLSTM(nn.Module):
    """
    Hybrid estimator that combines a lightweight MLP regressor with a
    classical LSTM for sequence tagging. When ``n_qubits`` is positive,
    the quantum LSTM implementation is expected to be provided in a
    separate module; the classical module keeps the interface
    compatible but raises a clear error if quantum functionality is
    requested.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # MLP regressor
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # LSTM
        if n_qubits > 0:
            # Quantum LSTM is not available in this classical module.
            self.lstm = None
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Tagging head
        self.tag_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, input_dim).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            reg_output: regression output shape (batch, 1)
            tag_output: tagging logits shape (batch, seq_len, 1)
        """
        # Regression on the first time step
        reg_out = self.regressor(x[:, 0])

        # Sequence tagging
        if self.lstm is None:
            raise NotImplementedError(
                "Quantum LSTM mode requires the quantum module; "
                "use the qml_code implementation."
            )
        lstm_out, _ = self.lstm(x)
        tag_logits = self.tag_head(lstm_out)
        return reg_out, tag_logits

__all__ = ["HybridEstimatorQLSTM"]
