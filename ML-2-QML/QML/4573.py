"""Hybrid quantum-classical CNN classifier with quanvolution filter and quantum expectation head."""
import numpy as np
import qiskit
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumCircuitWrapper:
    """Parametrised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = qiskit.transpile(self.circuit, self.backend)
        qobj = qiskit.assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.squeeze().tolist()
        exp = ctx.circuit.run(np.array(thetas))
        out = torch.tensor(exp, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        grads = []
        for val in inputs.squeeze().tolist():
            right = ctx.circuit.run(np.array([val + ctx.shift]))[0]
            left = ctx.circuit.run(np.array([val - ctx.shift]))[0]
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that maps a scalar to a probability via quantum expectation."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)

class QuanvolutionFilter(nn.Module):
    """Quantum filter that applies a random circuit to 2×2 patches."""
    def __init__(self, kernel_size=2, threshold=0.5):
        super().__init__()
        n_qubits = kernel_size ** 2
        self.n_qubits = n_qubits
        self.threshold = threshold
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = 100
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(n_qubits, 2)
        self.circuit.measure_all()

    def run_patch(self, patch: np.ndarray) -> float:
        bind = {}
        for i, val in enumerate(patch):
            bind[self.theta[i]] = np.pi if val > self.threshold else 0
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
        counts = job.result().get_counts()
        counts_sum = sum(int(bit) * count for key, count in counts.items() for bit in key)
        return counts_sum / (self.shots * self.n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape (batch, 1, 28, 28)
        batch, _, h, w = x.shape
        assert h == w == 28, "Expected 28x28 input."
        patches = []
        for i in range(0, h - 1, 2):
            for j in range(0, w - 1, 2):
                patch = x[..., i:i+2, j:j+2].reshape(batch, -1)
                patch_np = patch.detach().cpu().numpy()
                values = []
                for b in range(batch):
                    values.append(self.run_patch(patch_np[b]))
                patches.append(torch.tensor(values, device=x.device))
        output = torch.stack(patches, dim=1)
        return output.view(batch, -1)

class HybridBinaryNet(nn.Module):
    """Hybrid CNN with quanvolution filter and quantum expectation head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fc1 = nn.Linear(4 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(
            n_qubits=self.fc3.out_features,
            backend=backend,
            shots=100,
            shift=np.pi / 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qfilter(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x).squeeze()
        return torch.stack((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryNet"]
