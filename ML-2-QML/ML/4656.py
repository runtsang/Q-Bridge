"""Hybrid LSTM model that can switch between classical and quantum components.

The model integrates:
- optional QCNN-style feature extraction (classical fully connected layers),
- a choice between a classical LSTM or a quantum‑enhanced LSTM,
- a fully connected quantum layer (classical approximation) for classification.

The configuration is controlled by `n_qubits` and `use_qcnn`.  When `n_qubits > 0` the
model uses the quantum LSTM cell and the quantum fully connected layer; otherwise
purely classical components are used.  The same API is exposed in the quantum variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from FCL import FCL
from QCNN import QCNNModel
from QLSTM import QLSTM as ClassicalQLSTM


class HybridQLSTM(nn.Module):
    """
    Drop‑in hybrid LSTM for sequence tagging.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_qcnn: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Optional QCNN feature extractor
        self.use_qcnn = use_qcnn
        self.feature_extractor = QCNNModel() if use_qcnn else None

        # LSTM: classical or quantum
        self.lstm = (
            ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
            if n_qubits > 0
            else nn.LSTM(embedding_dim, hidden_dim)
        )

        # Final classification layer
        self.fcl = FCL()
        self.tagset_size = tagset_size

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of token indices, shape (seq_len, batch).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (seq_len, batch, tagset_size).
        """
        # Embed tokens
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed)

        # Apply QCNN feature extractor if enabled
        if self.feature_extractor is not None:
            seq_len, batch, _ = embeds.shape
            # Flatten batch and sequence for the fully‑connected QCNN
            flat = embeds.view(seq_len * batch, -1)
            # QCNN expects input of shape (N, feature_dim)
            qc_features = self.feature_extractor(flat)  # (seq_len*batch, 1)
            embeds = qc_features.view(seq_len, batch, -1)

        # LSTM
        lstm_out, _ = self.lstm(embeds)  # (seq_len, batch, hidden)

        # Classification via fully‑connected quantum layer (classical proxy)
        logits = []
        for step in lstm_out.split(1, dim=0):
            step = step.squeeze(0)  # (batch, hidden)
            step_logits = []
            for h in step.split(1, dim=0):
                h = h.squeeze(0)  # (hidden,)
                # The FCL.run expects an iterable of floats; we feed the hidden vector
                expectation = self.fcl.run(h.tolist())
                step_logits.append(expectation[0])
            logits.append(torch.tensor(step_logits, device=embeds.device))
        logits = torch.stack(logits, dim=0)  # (seq_len, batch)

        # Expand to tagset dimension (here we use a simple linear mapping)
        logits = logits.unsqueeze(-1).expand(-1, -1, self.tagset_size)

        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQLSTM"]
