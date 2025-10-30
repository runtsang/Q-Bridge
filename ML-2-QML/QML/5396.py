import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating quantum convolution steps."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class QFCModel(nn.Module):
    """CNN followed by a fully connected projection to four features."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

class QuantumCircuitWrapper(nn.Module):
    """
    Simple variational circuit that maps a scalar input to an expectation
    value of Pauli‑Z on a single qubit. The circuit is executed on an Aer
    simulator and returns a tensor of expectation values.
    """
    def __init__(self, backend: qiskit.providers.BaseBackend = None, shots: int = 1024) -> None:
        super().__init__()
        self.backend = backend if backend is not None else AerSimulator()
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(qiskit.circuit.Parameter('theta'), 0)
        self.circuit.measure_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1)
        outputs = []
        for val in x.squeeze().tolist():
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots,
                            parameter_binds=[{self.circuit.parameters[0]: val}])
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            # compute expectation of Z: p(0) - p(1)
            p0 = result.get('0', 0) / self.shots
            p1 = result.get('1', 0) / self.shots
            exp_z = p0 - p1
            outputs.append(exp_z)
        return torch.tensor(outputs, device=x.device).unsqueeze(1)

class HybridQuantumClassifier(nn.Module):
    """
    Hybrid binary classifier that uses a classical QCNN+QFC feature extractor
    followed by a quantum variational head. The quantum head is a simple
    single‑qubit circuit executed on an Aer simulator.
    """
    def __init__(self) -> None:
        super().__init__()
        # Classical feature extractor
        self.cnn = QFCModel().features
        self.lin_to_qcnn = nn.Linear(16 * 7 * 7, 8)
        self.qcnn = QCNNModel()
        # Quantum head
        self.quantum_head = QuantumCircuitWrapper(backend=AerSimulator(), shots=1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W)
        features = self.cnn(x)
        flattened = features.view(features.size(0), -1)
        qcnn_input = self.lin_to_qcnn(flattened)
        qcnn_output = self.qcnn(qcnn_input)
        # qcnn_output shape (batch, 1)
        quantum_out = self.quantum_head(qcnn_output)
        # Convert expectation [-1,1] to probability via sigmoid
        probs = torch.sigmoid((quantum_out + 1) / 2)
        return torch.cat((probs, 1 - probs), dim=-1)
