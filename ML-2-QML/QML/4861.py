import pennylane as qml
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as EstimatorQiskit
import torch
import torch.nn as nn
import torch.quantum as tq
import torch.quantum.functional as tqf

class QuantumConvFilter:
    """Quantum‑based filter for a 2‑D kernel."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.dev = qml.device("default.qubit", wires=kernel_size**2)
    def _circuit(self, data: np.ndarray):
        @qml.qnode(self.dev)
        def circuit():
            for i, val in enumerate(data.flatten()):
                angle = np.pi if val > self.threshold else 0.0
                qml.RX(angle, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.kernel_size**2)]
        return circuit
    def run(self, data: np.ndarray) -> float:
        circuit = self._circuit(data)
        results = circuit()
        return np.mean(results)

class QuantumQCNN:
    """QCNN circuit built from parameterized layers similar to the QML seed."""
    def __init__(self, num_qubits: int = 8, depth: int = 3):
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit = self._build_circuit()
        self.backend = Aer.get_backend("qasm_simulator")
        self.estimator = EstimatorQiskit()
    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Feature map
        for i in range(self.num_qubits):
            qc.h(i)
        # Ansatz layers
        for d in range(self.depth):
            for i in range(self.num_qubits):
                qc.rx(np.random.uniform(0, 2*np.pi), i)
            for i in range(self.num_qubits - 1):
                qc.cx(i, i+1)
            qc.barrier()
        qc.measure_all()
        return qc
    def run(self, data: np.ndarray) -> np.ndarray:
        # In this simplified version, data is ignored; we just evaluate expectation
        result = self.estimator.run([self.circuit], [SparsePauliOp.from_list([("Z"*self.num_qubits, 1)])])
        return result[0].data

class QuantumQLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell using TorchQuantum."""
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
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
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
    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class QuantumHybridModel(nn.Module):
    """End‑to‑end quantum‑classical hybrid model."""
    def __init__(self, vocab_size: int, tagset_size: int, n_qubits_lstm: int = 4):
        super().__init__()
        self.conv = QuantumConvFilter(kernel_size=2, threshold=0.5)
        self.qcnn = QuantumQCNN(num_qubits=8, depth=3)
        self.lstm = QuantumQLSTM(input_dim=8, hidden_dim=8, n_qubits=n_qubits_lstm)
        self.classifier = nn.Linear(8, tagset_size)
        self.word_embeddings = nn.Embedding(vocab_size, 8)
    def forward(self, images: np.ndarray, sequence: torch.Tensor) -> torch.Tensor:
        # Quantum convolution
        batch_size = images.shape[0]
        conv_out = []
        for img in images:
            conv_out.append(self.conv.run(img))
        conv_out = torch.tensor(conv_out, dtype=torch.float32).to(self.lstm.device)
        # QCNN processing
        qcnn_out = []
        for val in conv_out:
            qcnn_out.append(self.qcnn.run(np.array([val])))
        qcnn_out = torch.tensor(qcnn_out, dtype=torch.float32).to(self.lstm.device)
        # Prepare for LSTM
        embeddings = self.word_embeddings(sequence)
        lstm_out, _ = self.lstm(embeddings, None)
        logits = self.classifier(lstm_out)
        return torch.log_softmax(logits, dim=1)

def Conv() -> QuantumConvFilter:
    """Factory returning the quantum ConvFilter."""
    return QuantumConvFilter(kernel_size=2, threshold=0.5)

__all__ = [
    "QuantumConvFilter",
    "QuantumQCNN",
    "QuantumQLSTM",
    "QuantumHybridModel",
    "Conv",
]
