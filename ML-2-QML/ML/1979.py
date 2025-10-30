import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

class QLSTM(nn.Module):
    """
    Hybrid classical‑quantum LSTM cell.

    The classical part still owns the linear transformation that
    maps input+hidden to a space of dimension ``n_qubits``.  The
    quantum part then applies a small variational circuit to that
    vector, producing a scalar gate value in ``[0, 1]`` (after a
    sigmoid).  The design allows us to experiment with *any*
    quantum gate set – the circuit is built from *continuous*
    rotation gates, and the param‑count is often > 1 for each
    qubit.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 gate_families: List[str] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self._gate_families = gate_families or ["rx", "ry", "rz"]

        # Linear maps to qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum modules: one per gate
        self.forget_gate = self._build_gate()
        self.input_gate = self._build_gate()
        self.update_gate = self._build_gate()
        self.output_gate = self._build_gate()

    def _build_gate(self):
        # Build a tiny variational circuit that outputs a single probability
        from torchquantum import QuantumModule, QuantumDevice, GeneralEncoder
        from torchquantum.functional import cnot, rx, ry, rz

        class Gate(QuantumModule):
            def __init__(self, n_wires: int, families: List[str]):
                super().__init__()
                self.n_wires = n_wires
                self.families = families
                # Encode inputs as rotations on each wire
                self.encoder = GeneralEncoder(
                    [
                        {"input_idx": [i], "func": f"{families[i % len(families)]}",
                         "wires": [i]}
                        for i in range(n_wires)
                    ]
                )
                # Trainable rotation parameters
                self.params = nn.ModuleList(
                    [rx(has_params=True, trainable=True) for _ in range(n_wires)]
                )
                self.measure = torchquantum.MeasureAll()
            def forward(self, x: torch.Tensor):
                qdev = QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                self.encoder(qdev, x)
                for i, gate in enumerate(self.params):
                    gate(qdev, wires=i)
                # entangle wires
                for i in range(self.n_wires - 1):
                    cnot(qdev, wires=[i, i + 1])
                # single‑wire measurement
                return self.measure(qdev)
        return Gate(self.n_qubits, self._gate_families)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def get_gate_params(self) -> Dict[str, torch.Tensor]:
        """Return a dictionary of the trainable parameters of each gate."""
        return {
            "forget": torch.cat([p for p in self.forget_gate.params]),
            "input": torch.cat([p for p in self.input_gate.params]),
            "update": torch.cat([p for p in self.update_gate.params]),
            "output": torch.cat([p for p in self.output_gate.params]),
        }

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the hybrid QLSTM.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 gate_families: List[str] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits, gate_families)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
