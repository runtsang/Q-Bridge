import torch
import torch.nn as nn
import numpy as np
import qiskit
from qiskit import Aer, execute, transpile, assemble
from qiskit.circuit.random import random_circuit
import torchquantum as tq

class QuantumConvFilter(tq.QuantumModule):
    """Quantum convolution filter that emulates the quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: int = 127, shots: int = 100) -> None:
        super().__init__()
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        # Build a random parameterised circuit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit once on a single data sample."""
        flat = data.reshape(1, self.n_qubits)
        param_binds = []
        for vals in flat:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(vals)}
            param_binds.append(bind)
        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = sum(int(key[i] for i in range(self.n_qubits)) * val for key, val in result.items())
        return counts / (self.shots * self.n_qubits)

class QuantumKernel(tq.QuantumModule):
    """TorchQuantum implementation of an RBF‑like kernel via state‑overlap."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = tq.QuantumModule(
            [
                {"gate": "ry", "wires": [0]},
                {"gate": "ry", "wires": [1]},
                {"gate": "ry", "wires": [2]},
                {"gate": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_wires), y: (n_wires,)
        results = []
        for xi in x:
            self.q_device.reset_states(1)
            self.ansatz(self.q_device, xi, None)
            self.ansatz(self.q_device, -y, None)
            results.append(torch.abs(self.q_device.states[0]))
        return torch.stack(results)

class HybridConvKernelNet(nn.Module):
    """Quantum‑enhanced hybrid network mirroring the classical architecture."""
    def __init__(self, n_support: int = 10, n_wires: int = 4) -> None:
        super().__init__()
        # Classical front‑end (identical to the classical version)
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
        )
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
        )
        # Quantum kernel head
        self.kernel = QuantumKernel(n_wires=n_wires)
        # Support vectors (trainable parameters)
        self.support_vectors = nn.Parameter(torch.randn(n_support, n_wires))
        self.proj = nn.Linear(n_support, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.feature_extractor(x)  # (batch, 84)
        # Project the 84‑dim embedding onto the quantum‑kernel space
        quantum_features = x[:, :self.kernel.n_wires]  # (batch, n_wires)
        sims = torch.stack([self.kernel(quantum_features, sv) for sv in self.support_vectors], dim=1)
        logits = self.proj(sims)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["QuantumConvFilter", "QuantumKernel", "HybridConvKernelNet"]
