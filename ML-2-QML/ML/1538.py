import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class HybridQLSTM(nn.Module):
    """
    Hybrid LSTM cell that blends classical linear gates with a depth‑controlled
    variational quantum circuit.  The ``mode`` argument controls which part
    receives gradients:
      * ``"classical"`` – only classical gates are trained,
      * ``"quantum"``   – only quantum gates are trained,
      * ``"mixed"``     – gradients from both are summed.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        depth: int = 1,
        mode: str = "mixed",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.mode = mode

        # Classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum variational gates
        self.forget_qgate = self._build_variational_gate()
        self.input_qgate = self._build_variational_gate()
        self.update_qgate = self._build_variational_gate()
        self.output_qgate = self._build_variational_gate()

        # Linear maps that generate parameters for the quantum gates
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits * depth)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits * depth)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits * depth)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits * depth)

        # Map quantum output (n_qubits) to hidden_dim
        self.qgate_out = nn.Linear(n_qubits, hidden_dim)

    def _build_variational_gate(self) -> nn.Module:
        """
        Builds a depth‑controlled variational quantum circuit that maps a
        classical vector to a quantum measurement vector of size ``n_qubits``.
        """
        import torchquantum as tq
        import torchquantum.functional as tqf

        class VarGate(tq.QuantumModule):
            def __init__(self, n_wires: int, depth: int):
                super().__init__()
                self.n_wires = n_wires
                self.depth = depth
                # Parameter vector: one RX per wire per depth layer
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
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Classical gate outputs
            f_c = torch.sigmoid(self.forget_linear(combined))
            i_c = torch.sigmoid(self.input_linear(combined))
            g_c = torch.tanh(self.update_linear(combined))
            o_c = torch.sigmoid(self.output_linear(combined))

            # Quantum gate outputs (size: hidden_dim)
            f_q = torch.sigmoid(
                self.qgate_out(
                    self.forget_qgate(self.forget_lin(combined))
                )
            )
            i_q = torch.sigmoid(
                self.qgate_out(
                    self.input_qgate(self.input_lin(combined))
                )
            )
            g_q = torch.tanh(
                self.qgate_out(
                    self.update_qgate(self.update_lin(combined))
                )
            )
            o_q = torch.sigmoid(
                self.qgate_out(
                    self.output_qgate(self.output_lin(combined))
                )
            )

            if self.mode == "classical":
                f, i, g, o = f_c, i_c, g_c, o_c
            elif self.mode == "quantum":
                f, i, g, o = f_q, i_q, g_q, o_q
            else:  # mixed
                f, i, g, o = f_c + f_q, i_c + i_q, g_c + g_q, o_c + o_q

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class HybridTagger(nn.Module):
    """
    Sequence tagging model that can switch between a classical LSTM,
    a quantum LSTM, or the hybrid variant.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        depth: int = 1,
        mode: str = "mixed",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits == 0:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        else:
            self.lstm = HybridQLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits,
                depth=depth,
                mode=mode,
            )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "HybridTagger"]
