import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureSelector(nn.Module):
    """Mask‑based feature selector that keeps the most informative dimensions."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mask = nn.Parameter(torch.ones(in_features))
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weighted = x * self.mask
        topk_vals = torch.topk(weighted, self.out_features, dim=1).values
        return topk_vals

class HybridHead(nn.Module):
    """Differentiable quantum head that evaluates either a 1‑ or 2‑qubit Ry circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2, simple: bool = False):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self.simple = simple
        self._build_circuit()

    def _build_circuit(self):
        import qiskit
        from qiskit.circuit import QuantumCircuit, Parameter
        self.theta = Parameter("θ")
        self.circuit = QuantumCircuit(self.n_qubits)
        if self.simple:
            self.circuit.h(0)
            self.circuit.ry(self.theta, 0)
        else:
            self.circuit.h([0, 1])
            self.circuit.ry(self.theta, [0, 1])
        self.circuit.measure_all()

    def run_quantum(self, thetas: np.ndarray) -> np.ndarray:
        from qiskit import transpile, assemble
        compiled = transpile(self.circuit, self.backend)
        job = self.backend.run(assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.circuit.parameters[0]: theta} for theta in thetas]
        ))
        result = job.result().get_counts()
        if isinstance(result, list):
            return np.array([self._expectation(c) for c in result])
        return np.array([self._expectation(result)])

    def _expectation(self, count_dict):
        counts = np.array(list(count_dict.values()))
        states = np.array(list(count_dict.keys()), dtype=float)
        probs = counts / self.shots
        return np.dot(states, probs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        thetas = x.detach().cpu().numpy().flatten()
        exp = self.run_quantum(thetas)
        return torch.tensor(exp, dtype=torch.float32, device=x.device).unsqueeze(-1)

class QCNetExtended(nn.Module):
    """CNN backbone with optional feature selector and quantum head."""
    def __init__(self, num_classes: int = 2, use_selector: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.selector = FeatureSelector(55815, 2000) if use_selector else None

        import qiskit
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_head = HybridHead(2, backend, shots=200, simple=False)
        self.classifier = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        if self.selector is not None:
            x = self.selector(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        q_out = self.quantum_head(x)
        logits = self.classifier(q_out)
        return logits

__all__ = ["FeatureSelector", "HybridHead", "QCNetExtended"]
