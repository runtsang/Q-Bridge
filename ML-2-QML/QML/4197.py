import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
import torchquantum.functional as tqf
from typing import Tuple

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data into a parameterized quantum circuit."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated with a fixed TorchQuantum ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return a vector of kernel values, one per batch element.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states[:, 0])  # amplitude of |00..0> for each batch

class QLSTM(tq.QuantumModule):
    """LSTM cell where gates are realized by small quantum circuits."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class HybridKernelLSTM(nn.Module):
    """
    Quantum‑enhanced version of HybridKernelLSTM.
    The kernel is evaluated on a fixed quantum ansatz, while gates of the LSTM are realized
    by small quantum circuits.  Prototypes are learnable classical parameters and the final
    output is produced by a classical fully‑connected layer.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_prototypes: int = 5, n_qubits: int = 4) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        self.kernel = Kernel()
        self.lstm = QLSTM(input_dim, hidden_dim, n_qubits)
        self.fc = nn.Linear(hidden_dim, 1)

    def _kernel_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch*seq_len, input_dim)
        features = []
        for p in self.prototypes:
            k = self.kernel(x, p.unsqueeze(0))  # shape (batch*seq_len,)
            features.append(k.unsqueeze(1))
        return torch.cat(features, dim=1)  # (batch*seq_len, num_prototypes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        sequence: (batch, seq_len, input_dim)
        """
        batch, seq_len, _ = sequence.shape
        seq_flat = sequence.reshape(-1, sequence.size(-1))          # (batch*seq_len, D)
        features = self._kernel_features(seq_flat)                 # (batch*seq_len, P)
        features = features.reshape(batch, seq_len, -1)             # (batch, seq_len, P)
        # QLSTM expects inputs of shape (seq_len, batch, input_dim)
        lstm_out, _ = self.lstm(features.transpose(0, 1))
        last_hidden = lstm_out[-1, :, :]                            # (batch, hidden_dim)
        return self.fc(last_hidden)                                 # (batch, 1)
