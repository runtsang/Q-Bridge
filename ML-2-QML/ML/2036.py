import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable

class QLayerWrapper(nn.Module):
    """Wraps either a classical linear layer or a user supplied quantum gate.

    Parameters
    ----------
    input_dim : int
        Dimension of the concatenated input‑hidden vector.
    output_dim : int
        Dimension of the gate output (typically hidden_dim).
    depth : int, default 1
        Number of times the quantum gate is applied when ``use_quantum`` is True.
    use_quantum : bool, default False
        If True, ``quantum_gate`` must be supplied and will be called instead of a linear layer.
    quantum_gate : Optional[Callable[[torch.Tensor], torch.Tensor]], default None
        Quantum gate callable that accepts a tensor of shape ``(batch, input_dim)`` and returns
        a tensor of shape ``(batch, output_dim)``.
    dropout : float, default 0.0
        Dropout probability applied to the gate output.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int = 1,
        use_quantum: bool = False,
        quantum_gate: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.use_quantum = use_quantum
        if use_quantum:
            if quantum_gate is None:
                raise ValueError("quantum_gate must be provided when use_quantum is True")
            self.quantum_gate = quantum_gate
        else:
            self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            for _ in range(self.depth):
                x = self.quantum_gate(x)
        else:
            x = self.linear(x)
        return self.dropout(x)

class QLSTM(nn.Module):
    """Hybrid LSTM cell that can use classical or variational quantum gates.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input embeddings.
    hidden_dim : int
        Dimensionality of the hidden state.
    n_qubits : int, default 0
        Number of qubits to use in the quantum gates. If ``n_qubits`` is zero the cell
        behaves exactly like a conventional LSTM.
    depth : int, default 1
        Depth (number of repetitions) of each quantum gate when ``n_qubits`` > 0.
    dropout : float, default 0.0
        Dropout probability applied to each gate output.
    quantum_gate_factory : Optional[Callable[[str], Callable[[torch.Tensor], torch.Tensor]]], default None
        Factory that returns a quantum gate callable for a given gate name
        (``'forget'``, ``'input'``, ``'update'``, ``'output'``). When ``n_qubits`` > 0
        this factory must be supplied.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        *,
        depth: int = 1,
        dropout: float = 0.0,
        quantum_gate_factory: Optional[Callable[[str], Callable[[torch.Tensor], torch.Tensor]]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.dropout = dropout

        use_quantum = n_qubits > 0
        gate_dim = hidden_dim  # output of each gate

        if use_quantum:
            if quantum_gate_factory is None:
                raise ValueError("quantum_gate_factory must be provided when n_qubits > 0")
            self.forget_gate = QLayerWrapper(
                input_dim + hidden_dim,
                gate_dim,
                depth=depth,
                use_quantum=True,
                quantum_gate=quantum_gate_factory("forget"),
                dropout=dropout,
            )
            self.input_gate = QLayerWrapper(
                input_dim + hidden_dim,
                gate_dim,
                depth=depth,
                use_quantum=True,
                quantum_gate=quantum_gate_factory("input"),
                dropout=dropout,
            )
            self.update_gate = QLayerWrapper(
                input_dim + hidden_dim,
                gate_dim,
                depth=depth,
                use_quantum=True,
                quantum_gate=quantum_gate_factory("update"),
                dropout=dropout,
            )
            self.output_gate = QLayerWrapper(
                input_dim + hidden_dim,
                gate_dim,
                depth=depth,
                use_quantum=True,
                quantum_gate=quantum_gate_factory("output"),
                dropout=dropout,
            )
        else:
            # Classical gates
            self.forget_gate = QLayerWrapper(
                input_dim + hidden_dim,
                gate_dim,
                depth=depth,
                use_quantum=False,
                dropout=dropout,
            )
            self.input_gate = QLayerWrapper(
                input_dim + hidden_dim,
                gate_dim,
                depth=depth,
                use_quantum=False,
                dropout=dropout,
            )
            self.update_gate = QLayerWrapper(
                input_dim + hidden_dim,
                gate_dim,
                depth=depth,
                use_quantum=False,
                dropout=dropout,
            )
            self.output_gate = QLayerWrapper(
                input_dim + hidden_dim,
                gate_dim,
                depth=depth,
                use_quantum=False,
                dropout=dropout,
            )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the word embeddings.
    hidden_dim : int
        Hidden dimension of the LSTM.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of tags to predict.
    n_qubits : int, default 0
        Number of qubits to use in the LSTM. When zero a standard ``nn.LSTM`` is used.
    depth : int, default 1
        Depth of each quantum gate when ``n_qubits`` > 0.
    dropout : float, default 0.0
        Dropout probability applied to the gates.
    quantum_gate_factory : Optional[Callable[[str], Callable[[torch.Tensor], torch.Tensor]]], default None
        Factory that returns a quantum gate callable for a given gate name.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        *,
        depth: int = 1,
        dropout: float = 0.0,
        quantum_gate_factory: Optional[Callable[[str], Callable[[torch.Tensor], torch.Tensor]]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                depth=depth,
                dropout=dropout,
                quantum_gate_factory=quantum_gate_factory,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
