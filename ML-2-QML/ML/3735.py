import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RBFFunction(nn.Module):
    """Classical radial basis function kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class ClassicalKernel(nn.Module):
    """Wraps the RBF kernel for compatibility."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = RBFFunction(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()


class HybridQLSTM(nn.Module):
    """
    Hybrid LSTM model that optionally injects classical RBF kernel
    features into the input sequence before feeding it to a standard
    PyTorch LSTM cell.  The same class can be instantiated with
    ``use_kernel=False`` to operate purely classically.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        num_basis: int = 8,
        gamma: float = 1.0,
        use_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_kernel = use_kernel
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Kernel basis vectors (learnable)
        if self.use_kernel:
            self.kernel_basis = nn.Parameter(
                torch.randn(num_basis, embedding_dim), requires_grad=True
            )
            self.kernel = ClassicalKernel(gamma)

        # Effective input size
        input_size = embedding_dim + (num_basis if self.use_kernel else 0)
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: LongTensor of shape (seq_len,) containing token indices.
        Returns:
            Log‑softmax scores of shape (seq_len, tagset_size).
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, embedding_dim)
        if self.use_kernel:
            # Compute kernel features: (seq_len, num_basis)
            seq_len = embeds.size(0)
            basis = self.kernel_basis.expand(seq_len, -1, -1)  # (seq_len, num_basis, embedding_dim)
            emb_exp = embeds.unsqueeze(1).expand(-1, self.kernel_basis.size(0), -1)
            diff = emb_exp - basis
            kernel_feat = torch.exp(
                -self.kernel.gamma * torch.sum(diff * diff, dim=-1)
            )
            # Concatenate
            inputs = torch.cat([embeds, kernel_feat], dim=-1)
        else:
            inputs = embeds

        # LSTM expects (batch, seq_len, input_size) if batch_first=True
        inputs = inputs.unsqueeze(0)  # batch=1
        lstm_out, _ = self.lstm(inputs)  # (1, seq_len, hidden_dim)
        lstm_out = lstm_out.squeeze(0)  # (seq_len, hidden_dim)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM"]
