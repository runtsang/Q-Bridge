import pennylane as qml
import torch
from torch import nn
import numpy as np

# Quantum device with 4 qubits
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, interface="torch")
def quantum_feature_map(x: torch.Tensor) -> torch.Tensor:
    """
    Variational feature map that embeds a 4‑dimensional input vector
    into a 4‑qubit Hilbert space.  The circuit consists of
    parameterized RY gates followed by two CNOTs.  The returned
    expectation values of Pauli‑Z on each qubit serve as quantum
    features.
    """
    for i, xi in enumerate(x):
        qml.RY(xi, wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(4)])

class FraudDetectionHybrid(nn.Module):
    """Quantum‑enhanced fraud‑detection model.
    The quantum feature map is a small variational circuit that embeds
    the input vector into a 4‑qubit Hilbert space.  The output of the
    circuit (expectation values) is concatenated with the raw input
    and processed by a classical linear head.  The architecture
    mirrors the classical version while giving the quantum model
    access to non‑linear feature transformations.
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

        # Linear layer to process concatenated features
        concat_dim = input_dim + 4  # raw + quantum
        self.core = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
            )
            lstm_out_dim = lstm_hidden
        else:
            lstm_out_dim = hidden_dim

        self.classifier = nn.Linear(lstm_out_dim, 1)

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
            batch, feat = x.shape
            # Pad to 4 dimensions for quantum map
            pad = torch.zeros(batch, 4 - feat, device=x.device)
            x_q = torch.cat([x, pad], dim=1)
            # Compute quantum features sample‑wise
            qfeat = torch.stack([quantum_feature_map(x_q[i]) for i in range(batch)], dim=0)
            concat = torch.cat([x, qfeat], dim=1)
            out = self.core(concat)
            logits = self.classifier(out)
            if self.regression:
                risk = self.regressor(x)
                return logits, risk
            return logits
        elif x.dim() == 3:
            # Sequential case
            batch, seq, feat = x.shape
            out_seq = []
            for t in range(seq):
                xt = x[:, t, :]
                pad = torch.zeros(batch, 4 - feat, device=xt.device)
                xt_q = torch.cat([xt, pad], dim=1)
                qfeat = torch.stack([quantum_feature_map(xt_q[i]) for i in range(batch)], dim=0)
                concat = torch.cat([xt, qfeat], dim=1)
                out_t = self.core(concat)
                out_seq.append(out_t.unsqueeze(1))
            out_seq = torch.cat(out_seq, dim=1)
            out_seq, _ = self.lstm(out_seq)
            logits = self.classifier(out_seq)
            if self.regression:
                risk = self.regressor(x[:, -1, :])
                return logits, risk
            return logits
        else:
            raise ValueError("Input tensor must be 2 or 3 dimensional")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, *_ = self.forward(x)
        return torch.sigmoid(logits)

__all__ = ["FraudDetectionHybrid"]
