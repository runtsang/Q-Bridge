import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit.circuit.random import random_circuit
import numpy as np

class QuantumConvFilter(nn.Module):
    """Quantum convolution filter using Qiskit circuits."""
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, 1, kernel_size, kernel_size) or (kernel_size, kernel_size)
        Returns: Tensor of shape (batch,) with average probability of measuring |1>
        """
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        batch, _, H, W = x.shape
        outputs = []
        for i in range(batch):
            patch = x[i, 0].flatten().cpu().numpy()
            param_binds = {self.theta[j]: np.pi if patch[j] > self.threshold else 0 for j in range(self.n_qubits)}
            job = qiskit.execute(self.circuit,
                                 self.backend,
                                 shots=self.shots,
                                 parameter_binds=[param_binds])
            result = job.result().get_counts(self.circuit)
            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val
            avg_prob = counts / (self.shots * self.n_qubits)
            outputs.append(avg_prob)
        return torch.tensor(outputs, dtype=torch.float32, device=x.device)

class QuantumLSTM(nn.Module):
    """LSTM cell with quantum gates implemented via torchquantum."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
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
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
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

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
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

    def forward(self, inputs: torch.Tensor, states: tuple = None):
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

    def _init_states(self, inputs: torch.Tensor, states: tuple = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class ConvLSTM(nn.Module):
    """Hybrid quantum convolutional LSTM."""
    def __init__(self,
                 hidden_dim: int = 32,
                 num_layers: int = 1,
                 kernel_size: int = 2,
                 conv_threshold: float = 127,
                 n_qubits: int = 4,
                 batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first
        self.conv_filter = QuantumConvFilter(kernel_size=kernel_size,
                                             backend=qiskit.Aer.get_backend("qasm_simulator"),
                                             shots=100,
                                             threshold=conv_threshold)
        self.lstm = QuantumLSTM(input_dim=1,
                                hidden_dim=hidden_dim,
                                n_qubits=n_qubits)
        self.out_linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (seq_len, batch, 1, kernel_size, kernel_size) if batch_first=False
           or (batch, seq_len, 1, kernel_size, kernel_size) if batch_first=True
        Returns: Tensor of shape (seq_len, batch, 1) if batch_first=False
                 or (batch, seq_len, 1) if batch_first=True
        """
        if self.batch_first:
            batch, seq_len, C, H, W = x.shape
        else:
            seq_len, batch, C, H, W = x.shape

        conv_seq = []
        for t in range(seq_len):
            if self.batch_first:
                img = x[:, t]
            else:
                img = x[t]
            conv_out = self.conv_filter(img)  # (batch,)
            conv_out = conv_out.unsqueeze(1)  # (batch, 1)
            conv_seq.append(conv_out.unsqueeze(0))  # (1, batch, 1)

        conv_seq = torch.cat(conv_seq, dim=0)  # (seq_len, batch, 1)
        lstm_out, _ = self.lstm(conv_seq)
        out = self.out_linear(lstm_out)  # (seq_len, batch, 1)
        return out
