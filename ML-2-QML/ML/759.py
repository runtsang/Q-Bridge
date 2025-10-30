import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm

# --------------------------------------------------------------------------- #
#  Hybrid classical‑quantum LSTM
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    A hybrid LSTM that can run in either a fully classical mode or a quantum‑augmented mode.
    In classical mode the gates are standard linear layers with optional dropout and layer‑norm.
    In quantum mode the gates are realised by a compact variational circuit that outputs a
    probability amplitude for each gate, and the hidden state is updated by a
    classical linear transformation of the input and previous hidden state.
    The design permits fine‑grained control of the quantum depth and the number of qubits.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        dropout: float = 0.0,
        layer_norm: bool = False,
        qubit_depth: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = n_qubits > 0
        self.dropout = Dropout(dropout) if dropout else None
        self.layer_norm = LayerNorm(hidden_dim) if layer_norm else None

        # Classical linear gates
        gate_dim = hidden_dim
        self.forget_lin = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, gate_dim)

        if self.use_quantum:
            # Quantum‑augmented gates: 1‑qubit per gate type, with a depth‑2 variational block
            self.qgate = nn.ModuleDict({
                "forget": self._make_qgate(qubit_depth),
                "input":  self._make_qgate(qubit_depth),
                "update": self._make_qgate(qubit_depth),
                "output": self._make_qgate(qubit_depth),
            })
            # Linear mapping from input+hidden to qubits (pre‑quantum)
            self.lin_q = nn.Linear(input_dim + hidden_dim, n_qubits * 4)

    def _make_qgate(self, depth: int):
        """Build a 2‑depth parameter‑efficient quantum circuit."""
        import torchquantum as tq
        from torchquantum.functional import rx, ry, rzz, cnot

        class QGate(tq.QuantumModule):
            def __init__(self, n_wires: int, depth: int):
                super().__init__()
                self.n_wires = n_wires
                self.depth = depth
                self.params = nn.ParameterList(
                    [nn.Parameter(torch.randn(n_wires)) for _ in range(depth)]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (batch, n_wires)
                bsz = x.shape[0]
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
                # Encode classical data into rotation angles
                for i in range(self.n_wires):
                    rx(qdev, x[:, i], wires=[i])
                # Variational depth
                for d in range(self.depth):
                    for i in range(self.n_wires):
                        ry(qdev, self.params[d][i], wires=[i])
                    for i in range(self.n_wires - 1):
                        cnot(qdev, wires=[i, i + 1])
                return tq.measure_all(qdev, basis='z')

        return QGate(self.n_qubits, depth)

    def _apply_gate(self, gate_name: str, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            # Map classical vector to qubit amplitudes
            q_input = self.lin_q(x)
            q_input = q_input.view(-1, self.n_qubits * 4)
            gate_out = self.qgate[gate_name](q_input)
            # Convert back to probability (sigmoid)
            return torch.sigmoid(gate_out)
        else:
            linear = getattr(self, f"{gate_name}_lin")
            return torch.sigmoid(linear(x))

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        inputs: (seq_len, batch, input_dim)
        states: (hx, cx) each (batch, hidden_dim)
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = self._apply_gate("forget", combined)
            i = self._apply_gate("input", combined)
            g = torch.tanh(self.update_lin(combined))
            o = self._apply_gate("output", combined)
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            if self.layer_norm is not None:
                hx = self.layer_norm(hx)
            if self.dropout is not None:
                hx = self.dropout(hx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
#  Sequence tagging model
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between a classical LSTM and the hybrid
    quantum‑augmented LSTM.  The interface mirrors the original seed.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        *,
        n_qubits: int = 0,
        dropout: float = 0.0,
        layer_norm: bool = False,
        qubit_depth: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                n_qubits=n_qubits,
                dropout=dropout,
                layer_norm=layer_norm,
                qubit_depth=qubit_depth,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        sentence: (seq_len, batch) or (batch, seq_len)
        """
        # Ensure shape (seq_len, batch, embed)
        if sentence.dim() == 2:
            # (batch, seq_len) -> (seq_len, batch)
            sentence = sentence.transpose(0, 1)
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["HybridQLSTM", "LSTMTagger"]
