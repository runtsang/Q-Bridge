import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import Aer, assemble, transpile
import numpy as np

class QuantumKernelCircuit:
    """Parameterized circuit used as a quantum kernel."""
    def __init__(self, n_qubits: int, backend, shots: int = 1000):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
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

class SamplerQNN:
    """Quantum sampler constructed with Qiskit Machine Learning."""
    def __init__(self):
        from qiskit.circuit import ParameterVector
        from qiskit_machine_learning.neural_networks import SamplerQNN as QSampler
        from qiskit.primitives import StatevectorSampler as Sampler
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = qiskit.QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        sampler = Sampler()
        self.qsampler = QSampler(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

    def __call__(self, x):
        return self.qsampler(x)

class HybridClassifier(nn.Module):
    """Hybrid CNN + quantum kernel + sampler for binary classification."""
    def __init__(self, num_support: int = 10, n_qubits: int = 4, shots: int = 1000, gamma: float = 1.0):
        super().__init__()
        # Feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )
        # Quantum kernel engine
        self.backend = Aer.get_backend("aer_simulator")
        self.kernel_circuit = QuantumKernelCircuit(n_qubits, self.backend, shots)
        self.gamma = gamma
        # Learnable support vectors in feature space
        self.support_vectors = nn.Parameter(torch.randn(num_support, 1))
        # Linear head over kernel similarities
        self.fc = nn.Linear(num_support, 1)
        # Quantum sampler
        self.sampler = SamplerQNN()

    def kernel_similarity(self, features: torch.Tensor) -> torch.Tensor:
        """Compute quantum kernel similarities between batch features and support vectors."""
        batch_size = features.size(0)
        num_support = self.support_vectors.size(0)
        sims = torch.zeros(batch_size, num_support, device=features.device, dtype=features.dtype)
        for i in range(batch_size):
            for j in range(num_support):
                diff = float(features[i] - self.support_vectors[j])
                sims[i, j] = torch.tensor(self.kernel_circuit.run([diff])[0], device=features.device)
        return torch.exp(-self.gamma * sims.pow(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x).squeeze(-1)
        k = self.kernel_similarity(features)
        logits = self.fc(k)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridClassifier", "QuantumKernelCircuit", "SamplerQNN"]
