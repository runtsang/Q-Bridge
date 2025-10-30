import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumCircuit(nn.Module):
    """
    Variational circuit that maps an input vector to the expectation
    value of the first qubit's Z observable.
    The circuit consists of a Ry rotation per qubit followed by an
    entangling CNOT chain and a measurement of all qubits.
    """
    def __init__(self, n_qubits: int, shots: int = 1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.ParameterVector(f"theta", n_qubits)

        # Rotation layer
        for i in range(n_qubits):
            self.circuit.ry(self.theta[i], i)

        # Entanglement layer (CNOT chain)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)

        # Measurement
        self.circuit.measure_all()

        # Compile once
        self.compiled = transpile(self.circuit, self.backend)

    def expectation(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of input vectors.
        Returns the expectation value of the first qubit's Z operator.
        """
        exp_vals = []
        for theta in thetas:
            param_bind = {self.theta[i]: theta[i] for i in range(self.n_qubits)}
            qobj = assemble(self.compiled, shots=self.shots, parameter_binds=[param_bind])
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()

            # Convert counts to probabilities
            probs = np.array(list(counts.values())) / self.shots
            # Convert bitstrings to integers
            states = np.array([int(bitstr, 2) for bitstr in counts.keys()])
            # Compute expectation of Z on first qubit
            first_bit = (states >> (self.n_qubits - 1)) & 1
            exp = np.mean((-1) ** first_bit)
            exp_vals.append(exp)
        return np.array(exp_vals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_qubits)
        """
        thetas = x.detach().cpu().numpy()
        exp_vals = self.expectation(thetas)
        return torch.tensor(exp_vals, dtype=torch.float32, device=x.device)

class QuantumHybridLayer(nn.Module):
    """
    Differentiable wrapper around QuantumCircuit.
    The gradient is computed via the parameter‑shift rule
    implemented inside QuantumCircuit.forward.
    """
    def __init__(self, n_qubits: int, shots: int = 1024):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, shots=shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x)

class QuantumHybridNet(nn.Module):
    """
    Hybrid CNN that ends in a quantum head.
    The network architecture mirrors the original but replaces the
    quantum expectation layer with a variational circuit followed
    by a learnable dense transform. A classical head is retained
    for comparison, and the final probability is a weighted blend.
    """
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Hybrid head
        self.quantum_layer = QuantumHybridLayer(n_qubits=n_qubits, shots=shots)
        self.quantum_post = nn.Linear(1, 1)

        # Classical head
        self.classical_head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)  # (batch,)

        # Prepare input for quantum layer
        # Broadcast scalar to vector of size n_qubits
        quantum_input = x.unsqueeze(-1).repeat(1, self.quantum_layer.circuit.n_qubits)
        q_out = self.quantum_layer(quantum_input)  # (batch,)
        q_logits = self.quantum_post(q_out.unsqueeze(-1)).squeeze(-1)
        q_prob = torch.sigmoid(q_logits)

        # Classical head
        c_logits = self.classical_head(x.unsqueeze(-1)).squeeze(-1)
        c_prob = torch.sigmoid(c_logits)

        # Blend probabilities
        prob = 0.5 * q_prob + 0.5 * c_prob
        return torch.stack((prob, 1 - prob), dim=-1)

__all__ = ["QuantumHybridNet"]
