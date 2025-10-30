"""Hybrid quantum‑classical LSTM tagger with a quantum convolutional front‑end.

This module exposes a single :class:`HybridQLSTM_QCNN` class that
combines a classical embedding layer, a quantum convolutional neural
network (QCNN) for feature extraction, a quantum‑enhanced LSTM (QLSTM)
for sequence modelling, and a classical linear head for tag prediction.
The implementation is fully PyTorch‑based and is compatible with the
quantum modules defined in :mod:`hybrid_qml.qml_code`.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum modules. They are pure Python and can be imported
# without pulling in heavy quantum back‑ends at runtime.
# The actual quantum computation happens inside the submodules.
try:
    from.qml_code import QCNN, QLSTM
except Exception:  # pragma: no cover
    # In environments where the quantum back‑end is not available we
    # still expose a dummy interface so that the module can be imported.
    QCNN = None
    QLSTM = None


class HybridQLSTM_QCNN(nn.Module):
    """
    Hybrid quantum‑classical sequence tagger.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int
        Dimension of the word embeddings.
    hidden_dim : int
        Hidden size of the LSTM.
    tagset_size : int
        Number of possible output tags.
    n_qubits : int
        Number of qubits used in the quantum layers.  If ``0`` the
        quantum modules are replaced by classical equivalents.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Quantum convolutional front‑end
        if n_qubits > 0 and QCNN is not None:
            self.qcnn = QCNN(embedding_dim, n_qubits)
        else:
            # Fallback to a simple linear projection
            self.qcnn = nn.Linear(embedding_dim, embedding_dim)

        # Quantum LSTM
        if n_qubits > 0 and QLSTM is not None:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Classical tag head
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of word indices with shape ``(seq_len, batch)``.

        Returns
        -------
        torch.Tensor
            Log‑softmaxed tag logits with shape ``(seq_len, batch, tagset_size)``.
        """
        # 1. Embedding
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed_dim)

        seq_len, batch, embed_dim = embeds.shape

        # 2. Quantum convolutional feature extraction
        # Flatten the sequence so that QCNN can process each time step
        # independently.  The QCNN expects inputs of shape (batch, input_dim).
        embeds_flat = embeds.permute(1, 0, 2).reshape(batch * seq_len, embed_dim)

        if self.n_qubits > 0 and QCNN is not None:
            qcnn_out = self.qcnn(embeds_flat)  # (batch*seq_len, qcnn_dim)
        else:
            qcnn_out = self.qcnn(embeds_flat)  # Linear fallback

        # Reshape back to sequence format
        qcnn_out = qcnn_out.reshape(batch, seq_len, -1).permute(1, 0, 2)  # (seq_len, batch, qcnn_dim)

        # 3. Quantum LSTM
        if self.n_qubits > 0 and QLSTM is not None:
            lstm_out, _ = self.lstm(qcnn_out)
        else:
            lstm_out, _ = self.lstm(qcnn_out)

        # 4. Tag head
        tag_logits = self.hidden2tag(lstm_out)  # (seq_len, batch, tagset_size)

        return F.log_softmax(tag_logits, dim=2)

    def init_weights(self) -> None:
        """
        Initialise weights of the classical sub‑modules.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)


__all__ = ["HybridQLSTM_QCNN"]
