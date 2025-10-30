import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

from qml_hybrid import build_qcnn, build_qlstm

class HybridQCNNQLSTM(nn.Module):
    """
    Hybrid QCNN + LSTM model that merges classical feature extraction,
    a quantum convolution‑pooling block, and a switchable LSTM (classical
    or quantum).  The module is designed for multimodal inputs: an
    image‑like tensor and a token sequence.  The QCNN processes the
    image features, producing a scalar per time‑step that is concatenated
    with word embeddings before feeding into the LSTM.
    """

    def __init__(self,
                 image_dim: int,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 tagset_size: int,
                 n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical feature extractor: 4‑layer FC stack with residual
        self.feature_extractor = nn.Sequential(
            nn.Linear(image_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU()
        )

        # Reduce to 8 qubits for QCNN
        self.to_qcnn = nn.Linear(256, n_qubits)

        # Quantum convolution‑pooling block
        self.qcnn = build_qcnn(num_qubits=n_qubits, n_layers=3)

        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer: classical or quantum
        if n_qubits > 0:
            self.lstm = build_qlstm(
                input_dim=embedding_dim + 1,
                hidden_dim=hidden_dim,
                n_qubits=n_qubits
            )
        else:
            self.lstm = nn.LSTM(embedding_dim + 1, hidden_dim, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : torch.Tensor
            Tensor of shape (batch, seq_len, image_dim)
        tokens : torch.Tensor
            Tensor of shape (batch, seq_len) containing token indices
        Returns
        -------
        torch.Tensor
            Log‑softmax scores of shape (batch, seq_len, tagset_size)
        """
        batch, seq_len, _ = images.shape

        # Flatten time dimension for feature extraction
        flat_images = images.reshape(batch * seq_len, -1)
        feats = self.feature_extractor(flat_images)          # (batch*seq_len, 256)
        qcnn_input = self.to_qcnn(feats)                    # (batch*seq_len, n_qubits)
        qcnn_out = self.qcnn(qcnn_input)                    # (batch*seq_len, 1)
        qcnn_out = qcnn_out.reshape(batch, seq_len, -1)     # (batch, seq_len, 1)

        # Embed tokens
        embed = self.word_embeddings(tokens)                # (batch, seq_len, embedding_dim)

        # Concatenate quantum feature with embeddings
        combined = torch.cat([embed, qcnn_out], dim=-1)      # (batch, seq_len, embedding_dim+1)

        # LSTM processing
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(combined)
        else:
            lstm_out, _ = self.lstm(combined.transpose(0, 1))
            lstm_out = lstm_out.transpose(0, 1)

        logits = self.hidden2tag(lstm_out)                   # (batch, seq_len, tagset_size)
        return F.log_softmax(logits, dim=-1)
