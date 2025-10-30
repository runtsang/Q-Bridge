import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class VariationalCircuit:
    """Parameterised multi‑qubit circuit for a single‑qubit expectation.

    The circuit is composed of alternating layers of Ry/Rz rotations
    and CNOT entanglement.  The depth of the circuit is user
    configurable and the circuit is executed on a state‑vector
    simulator for exact gradients.
    """
    def __init__(self, n_qubits: int, depth: int, shots: int = 0):
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = AerSimulator(method='statevector')
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.params = []

        for d in range(depth):
            for q in range(n_qubits):
                theta = qiskit.circuit.Parameter(f"theta_{d}_{q}")
                self.params.append(theta)
                self._circuit.ry(theta, q)
                self._circuit.rz(theta, q)
            # entangle
            for q in range(n_qubits - 1):
                self._circuit.cx(q, q + 1)
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Run the circuit for each set of parameters and return
        the expectation value of Z on the first qubit.
        """
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots if self.shots > 0 else None,
            parameter_binds=[{p: t for p, t in zip(self.params, theta)} for theta in thetas]
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        if isinstance(counts, list):
            return np.array([self._expectation(c) for c in counts])
        return np.array([self._expectation(counts)])

    def _expectation(self, count_dict):
        """Compute <Z> on qubit 0 from a measurement count dictionary."""
        probs = {}
        total = 0
        for bitstring, cnt in count_dict.items():
            val = 1 if bitstring[-1] == '0' else -1  # qubit 0 is last bit
            probs[val] = probs.get(val, 0) + cnt
            total += cnt
        return probs.get(1, 0) / total - probs.get(-1, 0) / total

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit.

    The forward pass returns the expectation value of the circuit.
    The backward pass uses the parameter‑shift rule to compute exact
    gradients with respect to the input parameters.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        exp_val = circuit.run(thetas)[0]
        result = torch.tensor(exp_val, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        thetas = inputs.detach().cpu().numpy()
        grad_inputs = []
        for idx, theta in enumerate(thetas):
            theta_plus = thetas.copy()
            theta_minus = thetas.copy()
            theta_plus[idx] += shift
            theta_minus[idx] -= shift
            exp_plus = ctx.circuit.run(theta_plus)[0]
            exp_minus = ctx.circuit.run(theta_minus)[0]
            grad = (exp_plus - exp_minus) / (2 * np.sin(shift))
            grad_inputs.append(grad)
        grad_tensor = torch.tensor(grad_inputs, dtype=inputs.dtype, device=inputs.device)
        return grad_tensor * grad_output, None, None

class HybridLayer(nn.Module):
    """Hybrid layer that forwards a scalar through a quantum circuit."""
    def __init__(self, n_qubits: int, depth: int, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = VariationalCircuit(n_qubits, depth)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, 1)
        return HybridFunction.apply(x.squeeze(-1), self.circuit, self.shift)

class QCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = HybridLayer(n_qubits=1, depth=3, shift=np.pi/2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x).unsqueeze(-1)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["VariationalCircuit", "HybridLayer", "QCNet", "HybridFunction"]
