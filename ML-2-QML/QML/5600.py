import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumKernelAnsatz(tq.QuantumModule):
    """Encodes classical data via a sequence of Ry rotations."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for i in range(self.n_wires):
            params = x[:, i] if tq.op_name_dict["ry"].num_params else None
            func_name_dict["ry"](q_device, wires=[i], params=params)
        for i in reversed(range(self.n_wires)):
            params = -y[:, i] if tq.op_name_dict["ry"].num_params else None
            func_name_dict["ry"](q_device, wires=[i], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluated with a fixed ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernelAnsatz(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class QuantumQLSTM(tq.QuantumModule):
    """LSTM where each gate is a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
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
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                tgt = (wire + 1) % self.n_wires
                tq.cnot(qdev, wires=[wire, tgt])
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

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Applies a random two‑qubit circuit to 2×2 image patches."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumQCNN(tq.QuantumModule):
    """Quantum convolution‑pooling circuit followed by a classical classifier."""
    def __init__(self, n_qubits: int = 8):
        super().__init__()
        self.n_qubits = n_qubits
        self.feature_map = tq.ZFeatureMap(n_qubits)
        self.ansatz = tq.QuantumCircuit(n_qubits)
        for _ in range(2):
            self.ansatz.append(self._conv_layer(), range(n_qubits))
            self.ansatz.append(self._pool_layer(), range(n_qubits))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _conv_layer(self):
        qc = tq.QuantumCircuit(self.n_qubits)
        for i in range(0, self.n_qubits, 2):
            qc.ry(-np.pi/2, i)
            qc.cx(i, i+1)
            qc.ry(np.pi/2, i+1)
        return qc

    def _pool_layer(self):
        qc = tq.QuantumCircuit(self.n_qubits)
        for i in range(0, self.n_qubits, 2):
            qc.cx(i, i+1)
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(self.n_qubits, bsz=x.shape[0], device=x.device)
        self.feature_map(qdev, x)
        self.ansatz(qdev)
        return self.measure(qdev)

class HybridQuantumKernelModel:
    """Hybrid model that toggles between classical and quantum implementations."""
    def __init__(self, mode: str = 'classical', gamma: float = 1.0,
                 embedding_dim: int = 50, hidden_dim: int = 32,
                 vocab_size: int = 10000, tagset_size: int = 10,
                 n_qubits: int = 4):
        self.mode = mode
        if mode == 'classical':
            self.kernel = ClassicalKernel(gamma)
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, vocab_size, tagset_size)
            self.filter = QuanvolutionFilter()
            self.cnn = QCNNModel()
        else:
            self.kernel = QuantumKernel()
            self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits)
            self.filter = QuantumQuanvolutionFilter()
            self.cnn = QuantumQCNN()

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor):
        return torch.tensor([[self.kernel(x, y).item() for y in b] for x in a])

    def tag_sequence(self, sentence: torch.Tensor):
        return self.lstm.tag(sentence)

    def convolve(self, x: torch.Tensor):
        return self.filter(x)

    def encode(self, x: torch.Tensor):
        return self.cnn(x)
