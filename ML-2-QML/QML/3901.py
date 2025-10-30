import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator
import torchquantum as tq
import torchquantum.functional as tqf

class ParametricCircuit:
    def __init__(self, n_qubits: int, backend=None, shots: int = 512):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.theta = qiskit.circuit.Parameter('theta')
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        param_binds = [{self.theta: p} for p in params]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        if isinstance(counts, list):
            exp = [self._expectation(c) for c in counts]
        else:
            exp = [self._expectation(counts)]
        return np.array(exp)

    def _expectation(self, count_dict):
        probs = np.array(list(count_dict.values())) / self.shots
        states = np.array([int(k, 2) for k in count_dict.keys()])
        z_vals = np.where(states & 1, -1, 1)  # Z on first qubit
        return np.sum(z_vals * probs)

class QuantumHybridLayer(nn.Module):
    def __init__(self, n_qubits: int, backend=None, shots: int = 512, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = ParametricCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = torch.flatten(x)
        with torch.no_grad():
            exp = self.circuit.run(flat.cpu().numpy())
        return torch.tensor(exp, dtype=x.dtype, device=x.device)

class QuantumGate(tq.QuantumModule):
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
        dev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(dev, x)
        for i, gate in enumerate(self.params):
            gate(dev, wires=i)
        for i in range(self.n_wires - 1):
            tqf.cnot(dev, wires=[i, i + 1])
        tqf.cnot(dev, wires=[self.n_wires - 1, 0])
        return self.measure(dev)

class QuantumQLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.forget_gate = QuantumGate(n_qubits)
        self.input_gate = QuantumGate(n_qubits)
        self.update_gate = QuantumGate(n_qubits)
        self.output_gate = QuantumGate(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, seq: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(seq, states)
        outputs = []
        for x in seq.unbind(dim=1):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        return out, (hx, cx)

    def _init_states(self, seq, states):
        if states is not None:
            return states
        batch = seq.shape[0]
        device = seq.device
        return torch.zeros(batch, self.n_qubits, device=device), torch.zeros(batch, self.n_qubits, device=device)

class LSTMTaggerQuantum(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, tagset_size: int, n_qubits: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(sentence).unsqueeze(0)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out.squeeze(0))
        return F.log_softmax(logits, dim=1)

class HybridCNNQuantum(nn.Module):
    def __init__(self, head: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = head or QuantumHybridLayer(n_qubits=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

class HybridQuantumHybridNet:
    def __init__(self, vocab_size: int, tagset_size: int, n_qubits: int = 4):
        self.image_classifier = HybridCNNQuantum(head=QuantumHybridLayer(n_qubits))
        self.text_tagger = LSTMTaggerQuantum(vocab_size, embedding_dim=128, hidden_dim=256, tagset_size=tagset_size, n_qubits=n_qubits)

    def classify_image(self, img: torch.Tensor) -> torch.Tensor:
        return self.image_classifier(img)

    def tag_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        return self.text_tagger(seq)
