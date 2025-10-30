import torch
import torch.nn as nn
import torch.nn.functional as F
from.QCNN import QCNNModel

class HybridTagger(nn.Module):
    """
    Classical hybrid tagger that combines a fully‑connected QCNN
    feature extractor with a standard LSTM encoder.  It is a
    drop‑in replacement for the original QLSTM/LSTMTagger pair.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        embedding_dim: int = 8,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        # Fixed embedding size to match QCNN input dimension
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.qcnn = QCNNModel()
        # LSTM receives the scalar QCNN output
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.LongTensor
            Shape (seq_len, batch) containing token indices.

        Returns
        -------
        torch.Tensor
            Log‑softmaxed tag logits of shape (seq_len, batch, tagset_size).
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, 8)
        seq_len, batch, _ = embeds.shape
        features = []
        for i in range(seq_len):
            # QCNN expects (batch, 8)
            out = self.qcnn(embeds[i])  # (batch, 1)
            features.append(out.unsqueeze(0))
        seq_features = torch.cat(features, dim=0)  # (seq_len, batch, 1)
        lstm_out, _ = self.lstm(seq_features)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["HybridTagger"]
