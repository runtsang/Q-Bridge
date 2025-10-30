import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple

class QLSTM(nn.Module):
    """Hybrid LSTM that can delegate to a quantum cell or fall back to a classical LSTM.

    Parameters
    ----------
    embedding_dim : int
        Size of the word embeddings.
    hidden_dim : int
        Hidden state dimensionality.
    vocab_size : int
        Vocabulary size.
    tagset_size : int
        Number of output tags.
    n_qubits : int, optional
        Number of qubits in the quantum LSTM cell. If zero, a pure
        ``nn.LSTM`` is used.
    quantum_cell : Callable[[torch.Tensor], Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]], optional
        A callable that implements the quantum LSTM cell. Must expose the
        same interface as :class:`torch.nn.Module` and accept the
        concatenated input+hidden vector. This keeps the classical
        implementation free of quantum dependencies.

    Notes
    -----
    Hidden states can be regularised via a graph constructed from their
    pairwise fidelities.  The adjacency matrix is used to smooth the
    hidden representation across time steps.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        quantum_cell: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            if quantum_cell is None:
                raise ValueError(
                    "Quantum cell must be supplied when n_qubits > 0."
                )
            self.lstm = quantum_cell
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Return log‑softmax tag logits for a sequence."""
        embeds = self.word_embeddings(sentence).unsqueeze(1)  # (seq_len, 1, emb)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)

    # --- graph utilities ----------------------------------------------------
    @staticmethod
    def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Squared overlap of two unit‑norm tensors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def build_adjacency(self, hidden_states: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
        """Construct a weighted adjacency matrix from hidden state fidelities."""
        seq_len = hidden_states.size(0)
        adj = torch.zeros(seq_len, seq_len, device=hidden_states.device)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                fid = self._state_fidelity(hidden_states[i], hidden_states[j])
                if fid >= threshold:
                    adj[i, j] = adj[j, i] = 1.0
                elif fid >= threshold * 0.5:
                    adj[i, j] = adj[j, i] = 0.5
        return adj

    def smooth_hidden(self, hidden_states: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Apply graph‑based smoothing to hidden states."""
        row_sums = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
        norm_adj = adj / row_sums
        return torch.matmul(norm_adj, hidden_states)

class EstimatorNN(nn.Module):
    """Simple feed‑forward regressor used when no quantum estimator is available."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

__all__ = ["QLSTM", "EstimatorNN"]
