"""Hybrid classical-quantum LSTM module for sequence tagging and regression.

This module implements a flexible LSTM-based architecture that can operate in two
modes:

* Tagging – uses a standard nn.Embedding + nn.LSTM + linear tag head.
* Regression – uses the same LSTM but with a single linear regression head.

The class accepts an ``n_qubits`` parameter to maintain API compatibility with the
quantum version.  When ``n_qubits > 0`` the flag is simply stored – the classical
implementation does not use quantum gates but the interface remains identical.

The module is intentionally lightweight and can be trained with any standard
PyTorch optimizer.  It is designed to be drop‑in compatible with the original
``QLSTM`` example while providing a seamless switch to a quantum backend.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQLSTM(nn.Module):
    """
    Classical LSTM with optional quantum‑inspired API.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features (e.g. word embedding size).
    hidden_dim : int
        Hidden size of the LSTM.
    n_qubits : int, default 0
        Number of qubits for the quantum backend.  Not used in the classical
        implementation but kept for API compatibility.
    task : str, {'tagging','regression'}, default 'tagging'
        Mode of operation: sequence tagging or regression.
    vocab_size : int, optional
        Size of the vocabulary; required only for ``tagging``.
    tagset_size : int, optional
        Number of tag classes; required only for ``tagging``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        task: str = "tagging",
        vocab_size: Optional[int] = None,
        tagset_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        if task == "tagging":
            if vocab_size is None or tagset_size is None:
                raise ValueError("vocab_size and tagset_size must be provided for tagging")
            self.word_embeddings = nn.Embedding(vocab_size, input_dim)
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, tagset_size)
        elif task == "regression":
            # In regression mode we treat the input as a sequence of feature vectors
            # (e.g. from a superposition data generator).  The LSTM reads the
            # sequence and the final hidden state is passed through a regression head.
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, 1)
        else:
            raise ValueError(f"Unsupported task {task!r}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            * For tagging: [seq_len, batch_size] LongTensor of token indices.
            * For regression: [seq_len, batch_size, input_dim] FloatTensor of features.

        Returns
        -------
        torch.Tensor
            * Tagging: log‑softmax logits of shape [seq_len, batch_size, tagset_size].
            * Regression: predicted values of shape [batch_size].
        """
        if self.task == "tagging":
            embeds = self.word_embeddings(inputs)
            lstm_out, _ = self.lstm(embeds)
            logits = self.head(lstm_out)
            return F.log_softmax(logits, dim=-1)
        else:  # regression
            lstm_out, _ = self.lstm(inputs)
            # Use the last hidden state for regression
            h_last = lstm_out[:, -1, :]
            return self.head(h_last).squeeze(-1)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper to initialize hidden and cell states.

        Parameters
        ----------
        batch_size : int
        device : torch.device

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (h_0, c_0)
        """
        h_0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return h_0, c_0

__all__ = ["HybridQLSTM"]
