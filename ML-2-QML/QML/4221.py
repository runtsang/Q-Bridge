"""Quantum hybrid classifier that combines a data‑uploading ansatz,
a quantum kernel, and optionally a quantum LSTM.
The module is a drop‑in replacement for the classical version."""
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np

class QuantumGateLayer(tq.QuantumModule):
    """Small variational block used inside the quantum LSTM."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for idx, wire in enumerate(range(self.n_wires)):
            qdev.rx(self.params[idx], wires=wire)
        for wire0, wire1 in [(i, i + 1) for i in range(self.n_wires - 1)] + [(self.n_wires - 1, 0)]:
            qdev.cz(wire0, wire1)
        return self.measure(qdev)

class QuantumLSTM(tq.QuantumModule):
    """Quantum LSTM cell where each gate is a small quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # gates
        self.forget_gate = QuantumGateLayer(n_qubits)
        self.input_gate = QuantumGateLayer(n_qubits)
        self.update_gate = QuantumGateLayer(n_qubits)
        self.output_gate = QuantumGateLayer(n_qubits)

        # linear projections to qubit space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch, seq_len, input_dim)
        """
        batch, seq_len, _ = inputs.shape
        hx = torch.zeros(batch, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch, self.hidden_dim, device=inputs.device)
        outputs = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            combined = torch.cat([x_t, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        return torch.cat(outputs, dim=1), (hx, cx)

class QuantumHybridClassifier(tq.QuantumModule):
    """
    Quantum counterpart of the hybrid architecture.
    Parameters
    ----------
    num_qubits : int
        Number of qubits for the data‑encoding ansatz.
    depth : int, default 3
        Depth of the variational circuit.
    use_lstm : bool, default False
        When True a quantum LSTM processes sequential data.
    """
    def __init__(self,
                 num_qubits: int,
                 depth: int = 3,
                 use_lstm: bool = False):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_lstm = use_lstm

        # data‑uploading ansatz with incremental encoding
        self.encoding = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(num_qubits)]
        )
        self.weights = nn.Parameter(torch.randn(num_qubits * depth))
        self.cnot_pattern = [(i, i + 1) for i in range(num_qubits - 1)] + [(num_qubits - 1, 0)]

        # quantum kernel
        self.kernel_ansatz = tq.QuantumModule()
        self.kernel_ansatz.__class__ = type(
            "KernelAnsatz",
            (tq.QuantumModule,),
            {
                "forward": self._kernel_forward
            }
        )()
        self.kernel_ansatz.n_wires = num_qubits
        self.kernel_ansatz.q_device = tq.QuantumDevice(n_wires=num_qubits)

        # optional quantum LSTM
        if use_lstm:
            self.lstm = QuantumLSTM(num_qubits, num_qubits, num_qubits)
        else:
            self.lstm = None

        # measurement for classification
        self.measure = tq.MeasureAll(tq.PauliZ)
        # output linear layer
        self.output_layer = nn.Linear(num_qubits + 1, 2)

    def _kernel_forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor):
        """Encode two data points and compute overlap."""
        q_device.reset_states(x.shape[0])
        for i in range(self.num_qubits):
            q_device.rx(x[:, i], wires=i)
        for i in range(self.num_qubits):
            q_device.rx(y[:, i], wires=i)
        return torch.abs(q_device.states.view(-1)[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, feat) for static data or (batch, seq_len, feat) if use_lstm.
        Returns
        -------
        logits : torch.Tensor
            Shape (batch, 2)
        """
        if self.use_lstm:
            batch, seq_len, feat = x.shape
            lstm_in = x.reshape(batch * seq_len, feat)
            lstm_out, _ = self.lstm(lstm_in)
            feat = lstm_out.reshape(batch, seq_len, -1).mean(dim=1)  # aggregate over time
        else:
            feat = x  # (batch, feat)

        # data‑uploading circuit
        qdev = tq.QuantumDevice(n_wires=self.num_qubits, bsz=feat.shape[0], device=feat.device)
        self.encoding(qdev, feat)
        for idx, wire in enumerate(range(self.num_qubits)):
            qdev.ry(self.weights[idx], wires=wire)
        for wire0, wire1 in self.cnot_pattern:
            qdev.cz(wire0, wire1)
        enc_out = self.measure(qdev)  # (batch, num_qubits)

        # quantum kernel evaluation
        k = self.kernel_ansatz(self.kernel_ansatz.q_device, feat, feat)  # self‑kernel for simplicity
        k = k.unsqueeze(1)  # (batch, 1)

        # concatenate classical and quantum features
        combined = torch.cat([enc_out, k], dim=1)  # (batch, num_qubits+1)
        logits = self.output_layer(combined)
        return logits

__all__ = ["QuantumHybridClassifier"]
