import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class HybridQLSTM(nn.Module):
    """
    Classical LSTM cell with optional quantum gate integration.
    The class can operate in three modes:
        - 'classical': all gates are linear + activations (default).
        - 'quantum': all gates are quantum circuits (requires qml backend).
        - 'hybrid': only the input gate uses a quantum circuit.
    This design allows ablation studies on quantum contribution.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int = 0,
                 mode: str = "classical") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.mode = mode.lower()

        # Linear projections for classical gates
        self.fc_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.fc_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.fc_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gate modules (only used in 'quantum' or 'hybrid' modes)
        if self.mode in {"quantum", "hybrid"}:
            # Placeholder: quantum gates are implemented in the QML module.
            # In the classical module we raise a warning if quantum mode is requested.
            raise RuntimeError(
                "Quantum mode requires the QML implementation. "
                "Use the quantum module to instantiate a HybridQLSTM with mode='quantum' or 'hybrid'."
            )

    def _init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (h, c) initialized to zeros."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Classical forward pass of the LSTM cell.
        Parameters
        ----------
        inputs : torch.Tensor
            Input sequence of shape (seq_len, batch, input_dim).
        states : tuple, optional
            Initial hidden and cell states (h, c). If None, zeros are used.
        Returns
        -------
        outputs : torch.Tensor
            Sequence of hidden states of shape (seq_len, batch, hidden_dim).
        final_states : tuple
            Final hidden and cell states (h, c).
        """
        batch_size = inputs.size(1)
        device = inputs.device
        h, c = self._init_state(batch_size, device) if states is None else states

        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, h], dim=1)
            f = torch.sigmoid(self.fc_forget(combined))
            i = torch.sigmoid(self.fc_input(combined))
            g = torch.tanh(self.fc_update(combined))
            o = torch.sigmoid(self.fc_output(combined))
            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (h, c)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses HybridQLSTM in classical mode.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 mode: str = "classical") -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, mode=mode)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of word indices of shape (seq_len, batch).
        Returns
        -------
        log_probs : torch.Tensor
            Log-softmax probabilities over tags of shape (seq_len, batch, tagset_size).
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["HybridQLSTM", "LSTMTagger"]
