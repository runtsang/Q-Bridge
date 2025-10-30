import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator

class QuanvCircuit:
    """Quantum filter emulating a classical convolution."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

class QuantumFilterLayer(nn.Module):
    """Wraps QuanvCircuit to produce a feature map from image patches."""
    def __init__(self, kernel_size=2, threshold=0.5, backend=None, shots=100):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.circuit = QuanvCircuit(kernel_size, self.backend, self.shots, self.threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W). Use first channel only.
        B, C, H, W = x.shape
        x = x[:, 0, :, :]
        patches = x.unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1)
        # patches: (B, H-1, W-1, k, k)
        B, Hp, Wp, _, _ = patches.shape
        patches = patches.contiguous().view(B * Hp * Wp, self.kernel_size, self.kernel_size)
        patches_np = patches.cpu().numpy()
        results = []
        for patch in patches_np:
            results.append(self.circuit.run(patch))
        results = torch.tensor(results, dtype=torch.float32)
        results = results.view(B, Hp, Wp)
        return results.unsqueeze(1)  # (B,1,Hp,Wp)

class QuantumFC(nn.Module):
    """Quantum fully connected block producing 4 qubit outputs."""
    def __init__(self, n_wires: int = 4, backend=None, shots: int = 1024):
        super().__init__()
        self.n_wires = n_wires
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.random_layer = qiskit.circuit.library.RandomCircuit(n_wires, depth=2)
        self.rx0 = Parameter("rx0")
        self.ry0 = Parameter("ry0")
        self.rz0 = Parameter("rz0")
        self.crx0 = Parameter("crx0")

        self.circuit = qiskit.QuantumCircuit(self.n_wires)
        self.circuit.append(self.random_layer, range(self.n_wires))
        self.circuit.rx(self.rx0, 0)
        self.circuit.ry(self.ry0, 1)
        self.circuit.rz(self.rz0, 2)
        self.circuit.crx(self.crx0, 0, 2)
        self.circuit.measure_all()

    def run(self, params):
        param_binds = {self.rx0: params[0], self.ry0: params[1], self.rz0: params[2], self.crx0: params[3]}
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[param_binds])
        job = self.backend.run(qobj)
        result = job.result().get_counts(self.circuit)
        expectations = []
        for qubit in range(self.n_wires):
            exp = 0
            for key, val in result.items():
                bit = int(key[::-1][qubit])  # last bit corresponds to qubit 0
                exp += (1 if bit==1 else -1) * val
            expectations.append(exp / self.shots)
        return torch.tensor(expectations)

class QuantumHybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumFC, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.tolist())
        result = torch.tensor([expectation])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs) * ctx.shift
        gradients = []
        for idx, value in enumerate(inputs.tolist()):
            right = ctx.circuit.run([value + shift[idx]])
            left = ctx.circuit.run([value - shift[idx]])
            gradients.append(right - left)
        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class QuantumHybrid(nn.Module):
    def __init__(self, n_wires: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_fc = QuantumFC(n_wires, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(inputs, self.quantum_fc, self.shift)

class QCNetQML(nn.Module):
    """Hybrid quantum‑classical CNN with quantum convolution and fully connected layers."""
    def __init__(self) -> None:
        super().__init__()
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.filter = QuantumFilterLayer(kernel_size=2, backend=backend, shots=100, threshold=0.5)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)
        self.norm = nn.BatchNorm1d(4)
        self.fc4 = nn.Linear(4, 1)
        self.quantum_head = QuantumHybrid(n_wires=1, backend=backend, shots=100, shift=np.pi/2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.filter(inputs)                       # (B,1,H-1,W-1)
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
        x = self.norm(x)
        x = self.fc4(x)
        prob = self.quantum_head(x)
        return torch.cat((prob, 1 - prob), dim=-1)

# Alias for consistency with the classical implementation
QCNet = QCNetQML

__all__ = ["QuanvCircuit", "QuantumFilterLayer", "QuantumFC",
           "QuantumHybridFunction", "QuantumHybrid", "QCNetQML", "QCNet"]
