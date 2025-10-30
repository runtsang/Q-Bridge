import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HybridQLSTM", "QLSTM"]

# --------------------------------------------------------------------------- #
# Classical building blocks
# --------------------------------------------------------------------------- #
class SamplerQNN(nn.Module):
    """Lightweight classical sampler network mirroring the QML sampler."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


class QuanvolutionFilter(nn.Module):
    """2‑pixel classical convolution that imitates a quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)


class RegressionHead(nn.Module):
    """Simple regression head for sequence‑level predictions."""
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)


# --------------------------------------------------------------------------- #
# Main hybrid model
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    A hybrid LSTM‑based model that can operate in a fully classical mode
    or use quantum sub‑modules for gates, filtering or regression.

    Parameters
    ----------
    embedding_dim : int
        Size of the input embeddings.
    hidden_dim : int
        Hidden dimensionality of the LSTM.
    vocab_size : int
        Number of unique tokens in the vocab.
    tagset_size : int
        Number of output tags for sequence tagging.
    n_qubits : int, default=0
        If >0, the LSTM gates are replaced with quantum circuits.
    use_regression : bool, default=False
        Attach a regression head after the LSTM.
    num_wires : int, default=4
        Quantum circuit width for regression or quanvolution layers.
    use_quanvolution : bool, default=False
        Apply a quanvolution filter before the LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_regression: bool = False,
        num_wires: int = 4,
        use_quanvolution: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.use_quanvolution = use_quanvolution
        self.use_regression = use_regression
        self.use_quantum_lstm = n_qubits > 0

        # Optional quanvolution feature extractor
        self.feature_extractor = QuanvolutionFilter() if self.use_quanvolution else None

        # LSTM (classical) or placeholder for quantum gate replacement
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Optional regression head
        self.reg_head = RegressionHead(hidden_dim) if self.use_regression else None

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Long tensor of indices with shape (seq_len,) or (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Log‑probabilities for tags; concatenated with regression output
            if ``use_regression`` is True.
        """
        embeds = self.embedding(sentence)
        if self.feature_extractor is not None:
            embeds = self.feature_extractor(embeds)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        output = F.log_softmax(tag_logits, dim=-1)

        if self.reg_head is not None:
            # Use the mean over the sequence as a simple pooling
            reg_logits = self.reg_head(lstm_out.mean(dim=1))
            output = torch.cat([output, reg_logits], dim=-1)
        return output


# Alias for backward compatibility with the original QLSTM name
QLSTM = HybridQLSTM
