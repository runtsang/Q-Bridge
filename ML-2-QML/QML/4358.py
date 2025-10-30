import torch
import torch.nn as nn
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumKernel(tq.QuantumModule):
    """Quantum‑kernel based on a fixed TorchQuantum ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        result = torch.zeros(batch, batch, device=x.device)
        for i in range(batch):
            for j in range(batch):
                self.ansatz(self.q_device, x[i:i+1], y[j:j+1])
                result[i, j] = torch.abs(self.q_device.states.view(-1)[0])
        return result

class QuantumQLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell from the QLSTM seed."""
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

        def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(q_device, wires=[wire, 0])
                else:
                    tqf.cnot(q_device, wires=[wire, wire + 1])
            return self.measure(q_device)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
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

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class QuantumSelfAttention:
    """Quantum self‑attention block built with Qiskit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, x: torch.Tensor) -> torch.Tensor:
        batch, embed_dim = x.shape
        outputs = []
        for i in range(batch):
            rotation_params = x[i].cpu().numpy()
            entangle_params = np.zeros(self.n_qubits - 1)
            circuit = self._build_circuit(rotation_params, entangle_params)
            job = qiskit.execute(circuit, self.backend, shots=1024)
            counts = job.result().get_counts(circuit)
            probs = np.zeros(self.n_qubits)
            for bitstring, count in counts.items():
                idx = int(bitstring, 2)
                probs[idx] = count
            probs /= 1024
            outputs.append(probs)
        return torch.tensor(outputs, device=x.device)

class QuantumClassifier:
    """Variational quantum circuit that outputs class logits."""
    def __init__(self, num_qubits: int, depth: int):
        self.num_qubits = num_qubits
        self.depth = depth
        self.qr = QuantumRegister(num_qubits, "q")
        self.cr = ClassicalRegister(num_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.theta = np.random.randn(num_qubits, depth)

    def _build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.num_qubits):
            circuit.rx(x[i], i)
        for d in range(self.depth):
            for i in range(self.num_qubits):
                circuit.ry(self.theta[i, d], i)
            for i in range(self.num_qubits - 1):
                circuit.cz(i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        logits = []
        for i in range(batch):
            circuit = self._build_circuit(x[i].cpu().numpy())
            job = qiskit.execute(circuit, self.backend, shots=1024)
            counts = job.result().get_counts(circuit)
            probs = np.zeros(self.num_qubits)
            for bitstring, count in counts.items():
                idx = int(bitstring, 2)
                probs[idx] = count
            probs /= 1024
            logits.append(probs)
        logits = torch.tensor(logits, device=x.device)
        return torch.log_softmax(logits, dim=1)

class QuantumClassifierModel(nn.Module):
    """Hybrid quantum interface mirroring the classical model."""
    def __init__(self, num_features: int, depth: int, n_qubits: int):
        super().__init__()
        self.kernel = QuantumKernel()
        self.lstm = QuantumQLSTM(num_features, num_features, n_qubits)
        self.attention = QuantumSelfAttention(num_features)
        self.classifier = QuantumClassifier(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, feat = x.shape
        x_flat = x.reshape(batch, -1)
        k_matrix = self.kernel(x_flat, x_flat)
        lstm_out, _ = self.lstm(k_matrix.unsqueeze(0))
        attn_out = self.attention(lstm_out.squeeze(0))
        logits = self.classifier(attn_out)
        return logits

__all__ = ["QuantumClassifierModel"]
