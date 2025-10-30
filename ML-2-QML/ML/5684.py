import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridNATQLSTM(nn.Module):
    """Hybrid classical architecture combining a CNN encoder, a fully connected projection,
    and an LSTM for sequence tagging. The model can be used purely classically,
    optionally with quantum modules swapped in via subclassing or composition."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 cnn_features: int = 8,
                 fc_features: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        # CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_features, cnn_features * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # Fully connected projection (simulating quantum FC)
        self.fc = nn.Sequential(
            nn.Linear(cnn_features * 2 * 7 * 7, fc_features),
            nn.ReLU(),
            nn.Linear(fc_features, 4)  # output dimension matches original QFCModel
        )
        self.norm = nn.BatchNorm1d(4)

        # Embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM for sequence tagging
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Final classification layer
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: LongTensor of shape (seq_len, batch_size)
            image: FloatTensor of shape (batch_size, 1, H, W)
        Returns:
            Log probabilities over tags.
        """
        # CNN forward
        img_features = self.cnn(image)          # shape (batch_size, 4)
        img_out = self.norm(self.fc(img_features))  # shape (batch_size, 4)

        # Embedding
        embeds = self.word_embeddings(sentence)  # (seq_len, batch_size, embedding_dim)

        # LSTM forward
        lstm_out, _ = self.lstm(embeds)          # (seq_len, batch_size, hidden_dim)

        # tag logits
        tag_logits = self.hidden2tag(lstm_out)   # (seq_len, batch_size, tagset_size)

        # combine image features with tag logits
        tag_logits += img_out.unsqueeze(0)       # broadcast over sequence length

        return F.log_softmax(tag_logits, dim=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(hidden_dim={self.hidden_dim})"

__all__ = ["HybridNATQLSTM"]
