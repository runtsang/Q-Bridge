import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, Aer, transpile, assemble

class QuantumKernelExtractor:
    """Quantum kernel extractor using a parameter‑shifted circuit."""
    def __init__(self, n_qubits: int, shots: int = 1024, shift: float = np.pi/2):
        self.n_qubits = n_qubits
        self.shots = shots
        self.shift = shift
        self.backend = Aer.get_backend('aer_simulator')
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        qc.barrier()
        qc.ry(self.theta, range(self.n_qubits))
        qc.measure_all()
        return qc

    def _expectation(self, params: np.ndarray) -> float:
        compiled = transpile(self.circuit, self.backend)
        param_binds = [{self.theta: p} for p in params]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        return np.dot(states, probs)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Return a 1‑dimensional quantum feature for each sample."""
        return np.array([self._expectation([float(x)]) for x in data])

class HybridHead(nn.Module):
    """Hybrid head that outputs a probability."""
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

class QuantumHybridBinaryClassifier(nn.Module):
    """Hybrid binary classifier with a quantum kernel extractor."""
    def __init__(self,
                 in_features: int,
                 n_qubits: int = 2,
                 shots: int = 1024,
                 shift: float = np.pi/2):
        super().__init__()
        self.kernel_extractor = QuantumKernelExtractor(n_qubits, shots, shift)
        total_in = in_features + 1  # one quantum feature
        self.head = HybridHead(total_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            data = x.detach().cpu().numpy().squeeze()
            qfeat = self.kernel_extractor(data)
        qfeat_tensor = torch.tensor(qfeat, dtype=torch.float32, device=x.device).unsqueeze(-1)
        x_combined = torch.cat([x, qfeat_tensor], dim=-1)
        return self.head(x_combined)

__all__ = ["QuantumHybridBinaryClassifier"]
