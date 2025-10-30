import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class VariationalCircuit(nn.Module):
    """
    4‑qubit variational circuit with a single‑parameter encoding on the first qubit.
    The circuit is parameterised by a learnable weight matrix of shape (n_qubits, 3)
    corresponding to RX, RY, RZ rotations for each qubit.
    """
    def __init__(self, n_qubits: int = 4, dev: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(dev, wires=n_qubits)
        self.params = nn.Parameter(torch.randn(n_qubits, 3))

    def forward(self, input_angle: torch.Tensor) -> torch.Tensor:
        @qml.qnode(self.dev, interface="torch")
        def circuit():
            # Encode the scalar input on the first qubit
            qml.RY(input_angle, wires=0)
            for i in range(self.n_qubits):
                qml.RX(self.params[i, 0], wires=i)
                qml.RY(self.params[i, 1], wires=i)
                qml.RZ(self.params[i, 2], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        return circuit()

class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch tensors and the variational circuit.
    Implements a simple central‑difference approximation for the gradient.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # inputs shape: (batch,) or (batch, 1)
        values = inputs.squeeze().tolist()
        outputs = []
        for val in values:
            out = circuit.forward(torch.tensor(val, dtype=torch.float32))
            outputs.append(out.item())
        result = torch.tensor(outputs, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.squeeze().tolist():
            out_plus = ctx.circuit.forward(torch.tensor(val + shift, dtype=torch.float32))
            out_minus = ctx.circuit.forward(torch.tensor(val - shift, dtype=torch.float32))
            grads.append(out_plus.item() - out_minus.item())
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class HybridClassifier(nn.Module):
    """
    Quantum‑enhanced binary classifier that mirrors the classical head.
    The output of the variational circuit is mapped to [0, 1] via (exp + 1)/2.
    """
    def __init__(self, n_qubits: int = 4, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = VariationalCircuit(n_qubits)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze()
        exp_val = HybridFunction.apply(x, self.circuit, self.shift)
        probs = (exp_val + 1) / 2  # map from [-1,1] to [0,1]
        return torch.cat([probs, 1 - probs], dim=-1)
