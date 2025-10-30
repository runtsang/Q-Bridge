import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

class QuantumHybridHead(nn.Module):
    """
    Quantum expectation head that maps a scalar to a probability.
    Uses a single‑qubit parameterised circuit executed on Aer.
    """
    def __init__(self, backend, shots=1024):
        super().__init__()
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter('theta')
        self.circuit = qiskit.QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def forward(self, inputs: torch.Tensor):
        flat = inputs.view(-1).cpu().numpy()
        exp_vals = []
        for val in flat:
            bound_circuit = self.circuit.bind_parameters({self.theta: val})
            compiled = transpile(bound_circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            count0 = result.get('0', 0)
            count1 = result.get('1', 0)
            exp = (count0 - count1) / self.shots
            exp_vals.append(exp)
        exp_tensor = torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)
        prob_tensor = torch.sigmoid(exp_tensor)
        return prob_tensor

class QuantumQLSTM(nn.Module):
    """
    LSTM where each gate is realised by a small quantum circuit.
    """
    class QGate(tq.QuantumModule):
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

        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

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

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class HybridClassifierQLSTM(nn.Module):
    """
    Hybrid classifier that combines a convolutional quantum‑enhanced binary
    head with a quantum LSTM tagger.  The model accepts an image tensor
    and a sentence tensor and outputs a binary probability and tag
    log‑probabilities.
    """
    def __init__(self, image_channels=3, lstm_hidden_dim=128, lstm_layers=1,
                 vocab_size=1000, tagset_size=10, n_qubits=4):
        super().__init__()
        # CNN backbone
        self.conv1 = nn.Conv2d(image_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)  # binary head

        # Quantum binary head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_head = QuantumHybridHead(backend, shots=200)

        # Quantum LSTM tagger
        self.word_embeddings = nn.Embedding(vocab_size, 64)
        self.lstm = QuantumQLSTM(64, lstm_hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(lstm_hidden_dim, tagset_size)

    def forward(self, image: torch.Tensor, sentence: torch.Tensor):
        # Image pathway
        x = F.relu(self.conv1(image))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        binary_logits = self.fc3(x)
        binary_prob = self.quantum_head(binary_logits)

        # Sequence pathway
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        tag_log_probs = F.log_softmax(tag_logits, dim=-1)

        return binary_prob, tag_log_probs

__all__ = ["HybridClassifierQLSTM"]
