import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumCircuit:
    """
    A parameterised two‑qubit circuit on Pennylane's default.qubit device.
    The circuit consists of a Hadamard layer, a Ry rotation per qubit,
    an entangling CNOT, and a measurement of Z on the first qubit.
    """
    def __init__(self, n_qubits: int = 2, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)
        self.theta = qml.ParameterVector("theta", length=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(theta):
            for i in range(self.n_qubits):
                qml.Hadamard(i)
                qml.RY(theta[i], i)
            qml.CNOT(0, 1)
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of parameter vectors."""
        return self._circuit(thetas)

class HybridFunction(torch.autograd.Function):
    """
    Differentiable wrapper that forwards a tensor through a Pennylane QNode
    and implements the parameter‑shift rule for gradients.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        with torch.no_grad():
            expectation = circuit.run(inputs.cpu().numpy())
        result = torch.tensor(expectation, requires_grad=True)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = torch.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            plus = inputs[i] + shift
            minus = inputs[i] - shift
            exp_plus = ctx.circuit.run(plus.cpu().numpy())
            exp_minus = ctx.circuit.run(minus.cpu().numpy())
            grad_inputs[i] = (exp_plus - exp_minus) / (2 * shift)
        return grad_inputs * grad_output, None, None

class HybridQuantum(nn.Module):
    """
    Quantum hybrid head that maps a real‑valued feature to a single probability
    using a parameterised quantum circuit and a learnable shift.
    """
    def __init__(self, n_qubits: int = 2, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class QCNetQuantum(nn.Module):
    """
    Convolutional network followed by a quantum hybrid head.
    Mirrors the classical QCNet architecture but replaces the final dense head
    with a Pennylane‑backed quantum layer.
    """
    def __init__(self, n_qubits: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.quantum_head = HybridQuantum(n_qubits=n_qubits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.quantum_head(x).T
        return torch.cat((x, 1 - x), dim=-1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning class logits."""
        return self.forward(X)

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 10,
            lr: float = 1e-3, device: str = "cpu") -> None:
        """Simple training loop exposing a sklearn-like fit API."""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        X, y = X.to(device), y.to(device)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            logits = self.forward(X).squeeze(-1)
            loss = criterion(logits, y.float())
            loss.backward()
            optimizer.step()
            if epoch % (epochs // 5 + 1) == 0:
                print(f"Epoch {epoch+1}/{epochs} loss: {loss.item():.4f}")

__all__ = ["QuantumCircuit", "HybridFunction", "HybridQuantum", "QCNetQuantum"]
