import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit as QC, transpile, assemble
from qiskit.providers.aer import AerSimulator

class ResidualBlock(nn.Module):
    """A simple residual block to improve gradient flow for deeper nets."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)) + x)

class VariationalCircuit:
    """Four‑qubit parametrised circuit with a single‑parameter shift ansatz."""
    def __init__(self,
                 n_qubits: int = 4,
                 backend: AerSimulator = AerSimulator(),
                 shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend   = backend
        self.shots     = shots
        self.theta = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(n_qubits)]
        self.circuit = QC(n_qubits)
        # Simple entangling ansatz: Ry(theta) on each qubit followed by a CNOT chain
        for i, p in enumerate(self.theta):
            self.circuit.ry(p, i)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, angles: list[float]) -> float:
        """Execute the circuit for the supplied angles and return the expectation of Z on qubit 0."""
        compiled = transpile(self.circuit, self.backend)
        param_binds = [{p: a for p, a in zip(self.theta, angles)}]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = 0.0
        for bitstring, cnt in counts.items():
            # Qiskit returns bitstrings with the least‑significant bit first.
            qubit0 = int(bitstring[0])  # qubit 0 is the least‑significant bit
            exp += (1 - 2 * qubit0) * cnt
        return exp / self.shots

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the variational circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        angles = inputs.detach().cpu().numpy()
        out = []
        for a in angles:
            # Use a 4‑qubit circuit; only the first angle is varied, others are fixed at 0.
            out.append(circuit.run([a, 0.0, 0.0, 0.0]))
        out = torch.tensor(out, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        angles = inputs.detach().cpu().numpy()
        grads = []
        for a in angles:
            exp_plus  = ctx.circuit.run([a + shift, 0.0, 0.0, 0.0])
            exp_minus = ctx.circuit.run([a - shift, 0.0, 0.0, 0.0])
            grad = (exp_plus - exp_minus) / (2 * np.sin(shift))
            grads.append(grad)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through the variational circuit."""
    def __init__(self, n_qubits: int = 4, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = VariationalCircuit(n_qubits)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)

class QuantumHybridClassifier(nn.Module):
    """CNN followed by a quantum‑variational hybrid head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.residual = ResidualBlock(15)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(n_qubits=4, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.residual(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        logits = self.hybrid(x.squeeze())
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

    def fit(self,
            dataloader,
            optimizer,
            loss_fn=nn.BCELoss(),
            epochs: int = 5,
            device: str = "cpu") -> None:
        """Simple training loop for the quantum‑hybrid classifier."""
        self.to(device)
        self.train()
        for epoch in range(epochs):
            for data, target in dataloader:
                data, target = data.to(device), target.to(device).float()
                optimizer.zero_grad()
                probs = self(data)
                loss = loss_fn(probs, target)
                loss.backward()
                optimizer.step()

    def predict(self, data: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """Return class probabilities for the given data."""
        self.eval()
        with torch.no_grad():
            return self(data.to(device))
