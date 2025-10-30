import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit as QC, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumExpectationHead:
    """
    Parameterised quantum circuit that maps a vector of angles onto a quantum
    state and returns the expectation value of Z on the first qubit.

    The circuit is constructed once with symbolic parameters and then
    evaluated for each batch element via parameter binding.
    """
    def __init__(self, n_qubits: int, backend=None, shots: int = 100):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or AerSimulator()
        # Symbolic parameters
        self.theta = [QC.Parameter(f'theta_{i}') for i in range(n_qubits)]
        # Build template circuit
        self.template = QC(n_qubits)
        for i, th in enumerate(self.theta):
            self.template.ry(th, i)
        # Measure all qubits to obtain counts
        self.template.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a batch of parameter vectors.
        Returns the expectation value of Z on qubit 0 for each sample.
        """
        expectations = []
        for theta in thetas:
            bound_circuit = self.template.bind_parameters({th: val for th, val in zip(self.theta, theta)})
            compiled = transpile(bound_circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            result = self.backend.run(qobj).result()
            counts = result.get_counts()
            total = sum(counts.values())
            # qiskit uses little‑endian bit ordering; first qubit is the last bit
            p0 = sum(cnt for bit, cnt in counts.items() if bit[-1] == '0') / total
            p1 = sum(cnt for bit, cnt in counts.items() if bit[-1] == '1') / total
            expectations.append(p0 - p1)  # expectation of Z
        return np.array(expectations, dtype=np.float32)

class HybridQCNet(nn.Module):
    """
    Quantum‑augmented binary classifier that combines a CNN backbone,
    an RBF kernel embedding, and a quantum expectation head.

    The kernel vector (size = number of prototypes) is fed into a
    parameterised quantum circuit.  The expectation value of Z on the
    first qubit is interpreted as the class‑one logit.
    """
    def __init__(self,
                 num_prototypes: int = 10,
                 gamma: float = 1.0,
                 shift: float = 0.0,
                 shots: int = 100):
        super().__init__()
        # Convolutional backbone identical to the classical version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 84)  # feature dimension for kernel comparison

        # Kernel parameters
        self.num_prototypes = num_prototypes
        self.gamma = gamma
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, 84))
        self.shift = shift

        # Quantum expectation head
        self.quantum_head = QuantumExpectationHead(num_prototypes, shots=shots)

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        sq_norm = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * sq_norm)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # (batch, 84)

        # Compute kernel vector against prototypes
        k = self.rbf_kernel(x, self.prototypes)  # (batch, num_prototypes)

        # Convert kernel vector to numpy for quantum evaluation
        k_np = k.detach().cpu().numpy()
        # Run quantum circuit for each sample
        z_expect = self.quantum_head.run(k_np)  # (batch,)
        logits = torch.tensor(z_expect, device=x.device, dtype=x.dtype) + self.shift
        probs = torch.sigmoid(logits)
        return torch.stack([probs, 1 - probs], dim=1)

__all__ = ["HybridQCNet"]
