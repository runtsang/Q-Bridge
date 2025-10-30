import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QuantumHybridQLSTM(nn.Module):
    """
    Quantum LSTM cell with depth‑controlled variational gates.
    Each gate is a variational circuit that returns a vector of size ``n_qubits``.
    The hidden state size is matched to ``n_qubits`` for compatibility.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        # Linear layers mapping input+hidden to quantum parameters
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits * depth)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits * depth)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits * depth)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits * depth)

        # Variational quantum gates for each LSTM gate
        self.forget_qgate = self._build_variational_gate()
        self.input_qgate = self._build_variational_gate()
        self.update_qgate = self._build_variational_gate()
        self.output_qgate = self._build_variational_gate()

    def _build_variational_gate(self) -> tq.QuantumModule:
        class VarGate(tq.QuantumModule):
            def __init__(self, n_wires: int, depth: int):
                super().__init__()
                self.n_wires = n_wires
                self.depth = depth
                self.params = nn.Parameter(torch.randn(n_wires * depth))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch = x.size(0)
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=x.device)
                # Encode input data as RX rotations
                for i in range(self.n_wires):
                    params = x[:, i * self.depth:(i + 1) * self.depth]
                    tqf.rx(qdev, wires=i, params=params)
                # Entangling layer
                for i in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[i, i + 1])
                # Parameterized rotations
                for i in range(self.n_wires):
                    tqf.rx(qdev, wires=i, params=self.params[i * self.depth:(i + 1) * self.depth])
                return tq.measure_all(qdev)

        return VarGate(self.n_qubits, self.depth)

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

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(
                self.forget_qgate(self.forget_lin(combined))
            )
            i = torch.sigmoid(
                self.input_qgate(self.input_lin(combined))
            )
            g = torch.tanh(
                self.update_qgate(self.update_lin(combined))
            )
            o = torch.sigmoid(
                self.output_qgate(self.output_lin(combined))
            )
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class QuantumHybridTagger(nn.Module):
    """
    Sequence tagging model that uses the quantum LSTM cell.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QuantumHybridQLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits,
            depth=depth,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QuantumHybridQLSTM", "QuantumHybridTagger"]
