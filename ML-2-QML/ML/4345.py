import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class SamplerQNNHybrid(nn.Module):
    """
    Classical implementation of a hybrid sampler‑classifier.
    The model consists of:
    - A feed‑forward sampler network that maps input features to a 2‑dimensional probability vector.
    - An optional classical LSTM to process sequences of sampler outputs.
    - A fully‑connected classifier that maps the LSTM output to class logits.
    The architecture mirrors the quantum counterpart but uses pure PyTorch layers.
    """

    def __init__(
        self,
        input_dim: int = 2,
        sampler_hidden: int = 4,
        classifier_hidden: int = 4,
        classifier_output_dim: int = 2,
        use_qlstm: bool = False,
        lstm_hidden_dim: int = 8,
        lstm_n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.sampler = nn.Sequential(
            nn.Linear(input_dim, sampler_hidden),
            nn.Tanh(),
            nn.Linear(sampler_hidden, 2),
        )
        self.use_qlstm = use_qlstm
        if use_qlstm:
            self.lstm = nn.LSTM(
                input_size=2,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_n_layers,
                batch_first=True,
            )
        else:
            self.lstm = None
        self.classifier = nn.Sequential(
            nn.Linear(2 if not use_qlstm else lstm_hidden_dim, classifier_hidden),
            nn.ReLU(),
            nn.Linear(classifier_hidden, classifier_output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Output probabilities of shape (batch, seq_len, classifier_output_dim).
        """
        batch, seq_len, _ = x.shape
        # Flatten to process each time step independently
        x_flat = x.reshape(batch * seq_len, -1)
        sampler_out = torch.softmax(self.sampler(x_flat), dim=-1)
        sampler_out = sampler_out.reshape(batch, seq_len, 2)

        if self.lstm is not None:
            lstm_out, _ = self.lstm(sampler_out)
            features = lstm_out
        else:
            features = sampler_out

        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=-1)
        return probs
