import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQLSTM(nn.Module):
    """
    Classical hybrid model that can perform either sequence tagging or image classification.
    Parameters
    ----------
    task : str
        Either'sequence' for text or 'image' for image data.
    embedding_dim : int, optional
        Size of word embeddings (used only for sequence task).
    hidden_dim : int, optional
        Hidden size of the LSTM (used only for sequence task).
    vocab_size : int, optional
        Vocabulary size for the embedding layer (used only for sequence task).
    tagset_size : int, optional
        Number of output tags (used only for sequence task).
    """
    def __init__(self,
                 task: str,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 vocab_size: int = 10000,
                 tagset_size: int = 10) -> None:
        super().__init__()
        self.task = task
        if task == "sequence":
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        elif task == "image":
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
            )
            self.norm = nn.BatchNorm1d(4)
        else:
            raise ValueError("task must be'sequence' or 'image'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task == "sequence":
            # Expected shape: [batch, seq_len] or [seq_len, batch]
            if x.dim() == 2 and x.size(0) == self.word_embeddings.num_embeddings:
                # Assume [seq_len, batch] format
                embeds = self.word_embeddings(x)
                lstm_out, _ = self.lstm(embeds)
            else:
                # Assume [batch, seq_len] format
                embeds = self.word_embeddings(x)
                lstm_out, _ = self.lstm(embeds)
            logits = self.hidden2tag(lstm_out)
            return F.log_softmax(logits, dim=-1)
        else:  # image
            bsz = x.shape[0]
            features = self.features(x)
            flattened = features.view(bsz, -1)
            out = self.fc(flattened)
            return self.norm(out)

__all__ = ["HybridQLSTM"]
