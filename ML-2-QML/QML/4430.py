import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator
import torchquantum as tq
from typing import Sequence

class QuantumCircuitWrapper:
    """Aer‑based two‑qubit circuit with parameterised Ry gates."""
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        if backend is None:
            backend = AerSimulator()
        self.backend = backend
        self.shots = shots
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridQuantumHead(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int = 1, shift: float = np.pi / 2, shots: int = 100):
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, shots=shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape!= torch.Size([1, 1]):
            squeezed = torch.squeeze(inputs)
        else:
            squeezed = inputs[0]
        expectation_z = self.quantum_circuit.run(squeezed.tolist())
        return torch.tensor([expectation_z])

class QuanvolutionFilterQuantum(nn.Module):
    """Quantum quanvolution filter using TorchQuantum."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
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
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QCNNFeatureExtractor(nn.Module):
    """QCNN‑style fully‑connected stack (identical to the classical one)."""
    def __init__(self, in_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class KernelQuantum(tq.QuantumModule):
    """Quantum RBF kernel implemented with a fixed TorchQuantum ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.encoder(self.q_device, x)
        self.q_layer(self.q_device)
        self.encoder(self.q_device, y)
        self.q_layer(self.q_device)
        return torch.abs(self.q_device.states.view(-1)[0])

class QuanvolutionHybrid(nn.Module):
    """
    Quantum‑enhanced version of :class:`QuanvolutionHybrid`.  The filter and head
    are replaced by quantum counterparts while the convolutional backbone and
    QCNN feature extractor remain unchanged.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2, n_qubits: int = 1):
        super().__init__()
        self.filter = QuanvolutionFilterQuantum()
        # QCNet‑style backbone
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        combined_features_dim = 1 + out_channels * 14 * 14
        self.feature_extractor = QCNNFeatureExtractor(combined_features_dim)
        self.head = HybridQuantumHead(n_qubits, shift=np.pi / 2, shots=100)
        self.kernel = KernelQuantum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.filter(x)

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

        combined = torch.cat([x, f], dim=1)
        features = self.feature_extractor(combined)
        logits = self.head(features)
        return torch.cat((logits, 1 - logits), dim=-1)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuanvolutionHybrid", "QuanvolutionFilterQuantum", "QCNNFeatureExtractor", "HybridQuantumHead", "KernelQuantum"]
