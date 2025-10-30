import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class HybridFunction(torch.autograd.Function):
    """Differentiable quantum expectation head using parameter‑shift rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float, backend, shots: int):
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.backend = backend
        ctx.shots = shots
        params = inputs.detach().cpu().numpy()
        expectation = _run_circuit(circuit, params, backend, shots)
        result = torch.tensor(expectation, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        backend = ctx.backend
        shots = ctx.shots
        grad_inputs = np.zeros_like(inputs.numpy())
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                unit = np.zeros_like(inputs.numpy()[i])
                unit[j] = shift
                right = _run_circuit(circuit, inputs[i] + unit, backend, shots)
                left  = _run_circuit(circuit, inputs[i] - unit, backend, shots)
                grad_inputs[i, j] = (right - left) / 2
        grad_inputs = torch.tensor(grad_inputs, device=inputs.device, dtype=inputs.dtype)
        return grad_inputs * grad_output, None, None, None, None

def _run_circuit(circuit: QuantumCircuit, params: np.ndarray, backend, shots: int):
    """Execute a parametrised circuit and return expectation of Z."""
    if params.ndim == 1:
        params = params[np.newaxis, :]
    bound_circuits = []
    for p in params:
        bound = circuit.assign_parameters(dict(zip(circuit.parameters, p)), inplace=False)
        bound_circuits.append(bound)
    compiled = transpile(bound_circuits, backend)
    qobj = assemble(compiled, shots=shots)
    job = backend.run(qobj)
    result = job.result()
    expectations = []
    for circuit_result in result.results:
        counts = circuit_result.get_counts()
        exp = 0.0
        for bitstring, count in counts.items():
            outcome = 1 if bitstring.count('1') % 2 == 0 else -1
            exp += outcome * count
        expectations.append(exp / shots)
    return np.array(expectations)

class QuantumCircuitWrapper:
    """Minimal quantum circuit with parameter‑shift support."""
    def __init__(self, num_qubits: int, depth: int = 2, backend=None, shots: int = 1024):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        theta = ParameterVector("θ", self.num_qubits * self.depth)
        for i, qubit in enumerate(range(self.num_qubits)):
            qc.rx(theta[i], qubit)
        idx = self.num_qubits
        for _ in range(self.depth):
            for i in range(self.num_qubits):
                qc.ry(theta[idx], i)
                idx += 1
            for i in range(self.num_qubits - 1):
                qc.cz(i, i + 1)
        return qc

    def run(self, params: np.ndarray) -> np.ndarray:
        return _run_circuit(self.circuit, params, self.backend, self.shots)

class HybridClassifier(nn.Module):
    """Convolutional network with a quantum expectation head."""
    def __init__(self, n_qubits: int = 4, depth: int = 2, shots: int = 1024,
                 shift: float = np.pi / 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.quantum = QuantumCircuitWrapper(n_qubits, depth, shots=shots)
        self.shift = shift

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
        prob = HybridFunction.apply(x, self.quantum.circuit, self.shift,
                                     self.quantum.backend, self.quantum.shots)
        return torch.cat((prob, 1 - prob), dim=-1)

    def evaluate(self, inputs: torch.Tensor,
                 shots: int | None = None, seed: int | None = None) -> torch.Tensor:
        """Batch evaluation with optional shot noise."""
        self.eval()
        with torch.no_grad():
            probs = self.forward(inputs)
            if shots is None:
                return probs
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, 1/np.sqrt(shots), probs.shape)
            noisy = probs + torch.tensor(noise, device=probs.device, dtype=probs.dtype)
            return torch.clamp(noisy, 0, 1)

def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """Construct a feed‑forward classifier and metadata."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

__all__ = ["HybridClassifier", "build_classifier_circuit"]
