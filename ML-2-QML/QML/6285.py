import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import qiskit
import torchquantum as tq
import torchquantum.functional as tqf

class QuanvCircuit(nn.Module):
    """Quantum convolution filter that emulates the quantum quanvolution kernel."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.backend = qiskit.Aer.get_backend('qasm_simulator')
        self.shots = 100
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f'theta{i}') for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()
    def forward(self, x):
        # x shape: (batch, kernel_size, kernel_size)
        batch = x.shape[0]
        data = x.reshape(batch, self.n_qubits)
        param_binds = []
        for d in data:
            bind = {}
            for i, val in enumerate(d):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = []
        for key, cnt in counts.items():
            probs.append(cnt / self.shots)
        probs = np.array(probs).reshape(batch, -1)
        return torch.tensor(probs.mean(axis=1), dtype=torch.float32)

class QLayer(tq.QuantumModule):
    """Quantum circuit that replaces a linear gate."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)
    def forward(self, x):
        # x shape: (batch, n_wires)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for i, gate in enumerate(self.params):
            gate(qdev, wires=i)
        for i in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[i, i+1])
        return self.measure(qdev)

class QLSTM(nn.Module):
    """Hybrid LSTM where each gate is a quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget_gate = QLayer(n_qubits)
        self.input_gate = QLayer(n_qubits)
        self.update_gate = QLayer(n_qubits)
        self.output_gate = QLayer(n_qubits)
        self.fc_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.fc_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.fc_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.fc_output = nn.Linear(input_dim + hidden_dim, n_qubits)
    def forward(self, inputs, states=None):
        # inputs shape: (seq_len, batch, input_dim)
        batch_size = inputs.size(1)
        hx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        if states is not None:
            hx, cx = states
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.fc_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.fc_input(combined)))
            g = torch.tanh(self.update_gate(self.fc_update(combined)))
            o = torch.sigmoid(self.output_gate(self.fc_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class HybridConvQLSTM(nn.Module):
    """Drop‑in replacement for image‑based tagging that fuses a quantum convolution filter and a quantum LSTM."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5,
                 input_dim: int = 1, hidden_dim: int = 32,
                 n_qubits: int = 4, num_classes: int = 10):
        super().__init__()
        self.conv_filter = QuanvCircuit(kernel_size, threshold)
        self.lstm = QLSTM(input_dim, hidden_dim, n_qubits)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, images):
        # images shape: (batch, seq_len, 1, H, W)
        batch, seq_len, _, H, W = images.shape
        conv_out = []
        for t in range(seq_len):
            patch = images[:, t, :, :, :]  # (batch, 1, H, W)
            feat = self.conv_filter(patch)  # (batch,)
            conv_out.append(feat.unsqueeze(-1))
        conv_seq = torch.stack(conv_out, dim=1)  # (batch, seq_len)
        lstm_input = conv_seq.unsqueeze(-1).permute(1, 0, 2)
        lstm_out, _ = self.lstm(lstm_input)
        logits = self.classifier(lstm_out)
        return logits
