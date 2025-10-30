import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit as QC, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumPatchEncoder:
    """Quantum encoder for a 2×2 image patch using Ry gates and a random layer."""
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.theta = qiskit.circuit.Parameter('theta')
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = QC(self.n_qubits)
        qc.h(range(self.n_qubits))
        qc.ry(self.theta, range(self.n_qubits))
        # Random two‑qubit layer
        qc.cx(0, 1)
        qc.cx(2, 3)
        qc.cx(1, 2)
        qc.measure_all()
        return qc

    def run(self, thetas: np.ndarray):
        """Execute the circuit for a batch of angle values."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        # Expectation of Pauli‑Z measurement
        return np.array([self._expectation(count) for count in counts])

    def _expectation(self, count_dict):
        counts = np.array(list(count_dict.values()))
        states = np.array(list(count_dict.keys())).astype(float)
        probs = counts / self.shots
        return np.sum(states * probs)

class QuantumExpectationHead(nn.Module):
    """Hybrid layer that forwards activations through a parametrised quantum circuit."""
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.quantum_circuit = QuantumPatchEncoder(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Treat each element in the feature vector as a separate parameter
        batch, features = inputs.shape
        expectations = []
        for i in range(batch):
            row_expect = []
            for theta in inputs[i].detach().cpu().numpy():
                exp_val = self.quantum_circuit.run(np.array([theta]))
                row_expect.append(exp_val[0])
            expectations.append(row_expect)
        return torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)

class QuanvolutionFilterQuantum(nn.Module):
    """Quantum filter that encodes each 2×2 patch into a 4‑qubit state."""
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        super().__init__()
        self.encoder = QuantumPatchEncoder(n_qubits, backend, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        bsz = x.size(0)
        device = x.device
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2].reshape(bsz, -1)
                # Convert patch values to angles
                thetas = patch.cpu().numpy()
                expectations = self.encoder.run(thetas)
                patches.append(torch.tensor(expectations, dtype=x.dtype, device=device).unsqueeze(1))
        return torch.cat(patches, dim=1)

class QuanvolutionHybridClassifier(nn.Module):
    """Hybrid neural network that uses a quantum filter and a quantum expectation head."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10, shift: float = np.pi / 2):
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.linear = nn.Linear(4 * 14 * 14, 128)
        self.quantum_head = QuantumExpectationHead(4, shift=shift)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        probs = self.quantum_head(logits)
        logits = self.classifier(probs)
        return F.log_softmax(logits, dim=-1)
