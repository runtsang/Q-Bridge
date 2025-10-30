import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTM(nn.Module):
    """
    Classical LSTM cell with configurable gate activations and optional
    shared‑weight gates.  The public API matches the original seed
    (input_dim, hidden_dim, n_qubits) but the internal logic is enriched
    to allow experimentation without quantum dependencies.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dim : int
        Size of the hidden state.
    n_qubits : int
        Unused in the classical implementation but kept for API compatibility.
    gate_activation : str, optional
        Activation function for the forget, input and output gates.
        One of ``'sigmoid'`` (default), ``'tanh'`` or ``'softmax'``.
    shared_gates : bool, optional
        If True, all gates share a single linear projection.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int = 0,
                 gate_activation: str ='sigmoid',
                 shared_gates: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.gate_activation = gate_activation
        self.shared_gates = shared_gates

        if shared_gates:
            self.gate_linear = nn.Linear(input_dim + hidden_dim,
                                         4 * hidden_dim, bias=True)
        else:
            self.forget_linear = nn.Linear(input_dim + hidden_dim,
                                           hidden_dim, bias=True)
            self.input_linear = nn.Linear(input_dim + hidden_dim,
                                          hidden_dim, bias=True)
            self.update_linear = nn.Linear(input_dim + hidden_dim,
                                           hidden_dim, bias=True)
            self.output_linear = nn.Linear(input_dim + hidden_dim,
                                           hidden_dim, bias=True)

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_activation =='sigmoid':
            return torch.sigmoid(x)
        elif self.gate_activation == 'tanh':
            return torch.tanh(x)
        elif self.gate_activation =='softmax':
            return F.softmax(x, dim=-1)
        else:
            raise ValueError(f'Unsupported gate activation: {self.gate_activation}')

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Sequence of shape (seq_len, batch, input_dim).
        states : tuple of torch.Tensor or None
            Initial hidden and cell state.  If None, zeros are used.

        Returns
        -------
        outputs : torch.Tensor
            Hidden states for each time step (seq_len, batch, hidden_dim).
        (hx, cx) : tuple
            Final hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)  # (batch, input+hidden)

            if self.shared_gates:
                gates = self._gate(self.gate_linear(combined))
                f, i, g, o = gates.chunk(4, dim=-1)
            else:
                f = self._gate(self.forget_linear(combined))
                i = self._gate(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = self._gate(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can use the classical `QLSTM` defined above.
    The interface mirrors the original seed.

    Parameters
    ----------
    embedding_dim : int
        Size of word embeddings.
    hidden_dim : int
        Hidden state size of the LSTM.
    vocab_size : int
        Number of tokens in the vocabulary.
    tagset_size : int
        Number of target tags.
    n_qubits : int, optional
        Unused in the classical implementation but kept for API compatibility.
    gate_activation : str, optional
        Activation function for the gates in the underlying LSTM.
    shared_gates : bool, optional
        Whether the underlying LSTM shares gate parameters.
    """

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 gate_activation: str ='sigmoid',
                 shared_gates: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim,
                          hidden_dim,
                          n_qubits=n_qubits,
                          gate_activation=gate_activation,
                          shared_gates=shared_gates)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single sentence.

        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of word indices of shape (seq_len, batch).

        Returns
        -------
        log_probs : torch.Tensor
            Log‑probabilities for each tag at each position.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ['QLSTM', 'LSTMTagger']
