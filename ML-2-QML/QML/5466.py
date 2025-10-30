import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumCircuit(tq.QuantumModule):
    """Parametrised two‑qubit circuit executed on the Aer simulator."""
    def __init__(self, n_qubits: int, backend, shots: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        if isinstance(result, list):
            return torch.tensor([self._expectation(c) for c in result], dtype=torch.float32)
        return torch.tensor([self._expectation(result)], dtype=torch.float32)

    def _expectation(self, count_dict: dict) -> float:
        counts = np.array(list(count_dict.values()))
        states = np.array(list(count_dict.keys())).astype(float)
        probs = counts / self.shots
        return float(np.sum(states * probs))

class HybridQuantumLayer(tq.QuantumModule):
    """Wrapper that forwards a classical scalar through the quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantum.forward(x.tolist())

class QuantumKernel(tq.QuantumModule):
    """Fixed quantum kernel based on a random layer and Ry rotations."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.random_layer(self.q_device)
        for i in range(self.n_wires):
            tq.RY(x[i], wires=i)(self.q_device)
        for i in range(self.n_wires):
            tq.RY(-y[i], wires=i)(self.q_device)
        return torch.abs(self.q_device.states.view(-1)[0])

class QuantumHybridHead(tq.QuantumModule):
    """Hybrid head that combines a parametrised circuit with optional post‑processing."""
    def __init__(self, n_qubits: int, backend, shots: int):
        super().__init__()
        self.quantum_layer = HybridQuantumLayer(n_qubits, backend, shots, shift=np.pi / 2)
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.quantum_layer(x)
        return self.norm(out)

class HybridBinaryClassifier360(tq.QuantumModule):
    """Full hybrid binary classifier that mirrors the classical backbone and augments it with quantum heads."""
    def __init__(self, use_kernel: bool = True, n_qubits: int = 4, shots: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_head = QuantumHybridHead(n_qubits, backend, shots)

        self.use_kernel = use_kernel
        if use_kernel:
            self.kernel = QuantumKernel()
            self.register_buffer("support", torch.randn(10, n_qubits))

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

        # quantum head
        x = self.quantum_head(x)

        if self.use_kernel:
            batch = x.size(0)
            k = torch.zeros(batch, self.support.size(0), device=x.device)
            for i in range(batch):
                for j in range(self.support.size(0)):
                    k[i, j] = self.kernel(x[i], self.support[j])
            x = k.mean(dim=1, keepdim=True)

        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier360"]
