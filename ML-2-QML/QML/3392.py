import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

class QLSTM(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for gate, wire in zip(self.params, range(self.n_wires)):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class HybridEstimatorQLSTM(nn.Module):
    def __init__(self,
                 input_dim: int = 2,
                 hidden_sizes: Tuple[int,...] = (8, 4),
                 lstm_hidden_dim: int = 16,
                 vocab_size: int = 1000,
                 tagset_size: int = 10,
                 n_qubits: int = 4):
        super().__init__()
        # Quantum estimator circuit
        self.params = [Parameter(f"theta{i}") for i in range(input_dim)]
        self.qc = QuantumCircuit(input_dim)
        for i in range(input_dim):
            self.qc.ry(self.params[i], i)
        self.observable = SparsePauliOp.from_list([("Z" * input_dim, 1)])

        # Classical feed‑forward to map measurement to regression output
        layers = []
        in_dim = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.regressor = nn.Sequential(*layers)

        # Quantum LSTM tagger
        self.lstm = QLSTM(input_dim, lstm_hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(lstm_hidden_dim, tagset_size)

        # Embedding for sequence input
        self.embedding = nn.Embedding(vocab_size, input_dim)

    def forward(self,
                features: torch.Tensor,
                seq: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out = {}
        # Quantum estimator part
        batch = features.shape[0]
        exp_vals = []
        for i in range(batch):
            param_dict = {p: v.item() for p, v in zip(self.params, features[i])}
            state = Statevector.from_instruction(self.qc.bind_parameters(param_dict))
            exp_val = state.expectation_value(self.observable).real
            exp_vals.append(exp_val)
        exp_tensor = torch.tensor(exp_vals, dtype=torch.float32, device=features.device).unsqueeze(-1)
        out['regression'] = self.regressor(exp_tensor).squeeze(-1)

        # Quantum LSTM tagger
        if seq is not None:
            embeds = self.embedding(seq)  # (batch, seq_len, input_dim)
            embeds = embeds.permute(1, 0, 2)  # (seq_len, batch, input_dim)
            lstm_out, _ = self.lstm(embeds)  # (seq_len, batch, hidden_dim)
            lstm_out = lstm_out.permute(1, 0, 2)  # (batch, seq_len, hidden_dim)
            out['tags'] = F.log_softmax(self.hidden2tag(lstm_out), dim=-1)
        return out

__all__ = ["HybridEstimatorQLSTM"]
