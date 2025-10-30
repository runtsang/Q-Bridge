import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFCL_QLSTM(nn.Module):
    """
    Hybrid module combining a fully connected layer and an LSTM.
    Supports classical, quantum, or mixed operation modes.
    The quantum variants are implemented in the qml_code module.
    """
    def __init__(self, n_features: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 n_qubits_fcl: int = 0, n_qubits_lstm: int = 0, mode: str = "classical"):
        super().__init__()
        self.mode = mode
        self.n_qubits_fcl = n_qubits_fcl
        self.n_qubits_lstm = n_qubits_lstm

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, n_features)

        # Fully connected layer
        if self.mode == "classical" or (self.mode == "mixed" and self.n_qubits_fcl == 0):
            self.fcl = nn.Linear(n_features, 1)
        else:
            self.fcl = None  # quantum FCL placeholder

        # LSTM
        if self.mode == "classical" or (self.mode == "mixed" and self.n_qubits_lstm == 0):
            self.lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        else:
            self.lstm = None  # quantum LSTM placeholder

        # Output projection
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        """
        sentence: (batch_size, seq_len) LongTensor of token indices
        """
        # Embed
        embeds = self.embedding(sentence)  # (batch_size, seq_len, n_features)

        # LSTM
        if self.lstm is not None:
            lstm_out, _ = self.lstm(embeds)  # (batch_size, seq_len, hidden_dim)
        else:
            raise NotImplementedError("Quantum LSTM not available in classical implementation.")

        # Fully connected layer
        if self.fcl is not None:
            fcl_out = self.fcl(lstm_out)  # (batch_size, seq_len, 1)
        else:
            fcl_out = lstm_out

        # Output projection
        tag_logits = self.hidden2tag(fcl_out.squeeze(-1))  # (batch_size, seq_len, tagset_size)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["HybridFCL_QLSTM"]
