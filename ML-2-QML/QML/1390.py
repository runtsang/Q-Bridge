import pennylane as qml
import torch

class QuantumNATModel:
    """Quantum model built with Pennylane, 8 qubits, variational layer."""
    def __init__(self, n_qubits=8, shots=1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        self.params = torch.randn(n_qubits, 3, requires_grad=True)
        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RX(x[i] * self.params[i,0], wires=i)
                qml.RY(x[i] * self.params[i,1], wires=i)
                qml.RZ(x[i] * self.params[i,2], wires=i)
            for i in range(n_qubits-1):
                qml.CNOT(wires=[i, i+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.circuit = circuit
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_out = []
        for sample in x:
            sample_norm = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
            out = self.circuit(sample_norm)
            batch_out.append(out)
        return torch.stack(batch_out)
    def parameters(self):
        return [self.params]
    def zero_grad(self):
        if self.params.grad is not None:
            self.params.grad.zero_()
    def step(self, lr=0.01):
        self.params.data -= lr * self.params.grad

__all__ = ["QuantumNATModel"]
