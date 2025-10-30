import torch
import torch.nn as nn
import numpy as np
import qiskit
from qiskit import Aer, assemble, transpile

class QuantumCircuit:
    """Parametrised 4‑qubit circuit with a random layer and Pauli‑Z expectation output."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        # Encoding: one Ry per qubit
        self.enc_params = [qiskit.circuit.Parameter(f"enc_{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.ry(self.enc_params[i], i)
        # Random layer of 50 gates
        self._add_random_layer()
        # Measurement
        self.circuit.measure_all()

    def _add_random_layer(self):
        import random
        ops = [qiskit.circuit.gates.RxGate, qiskit.circuit.gates.RzGate,
               qiskit.circuit.gates.CNOTGate, qiskit.circuit.gates.CRXGate]
        for _ in range(50):
            op = random.choice(ops)
            if op in [qiskit.circuit.gates.RxGate, qiskit.circuit.gates.RzGate]:
                wire = random.randint(0, self.n_qubits-1)
                param = qiskit.circuit.Parameter(f"p{_}")
                self.circuit.append(op(param), [wire])
            elif op in [qiskit.circuit.gates.CNOTGate]:
                w1, w2 = random.sample(range(self.n_qubits), 2)
                self.circuit.append(op(), [w1, w2])
            elif op in [qiskit.circuit.gates.CRXGate]:
                w1, w2 = random.sample(range(self.n_qubits), 2)
                param = qiskit.circuit.Parameter(f"p{_}")
                self.circuit.append(op(param), [w1, w2])

    def run(self, angles: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of angle vectors."""
        results = []
        for angle in angles:
            # Bind encoding parameters
            param_dict = {self.enc_params[i]: angle[i] for i in range(self.n_qubits)}
            # Bind remaining parameters to zero
            for p in self.circuit.parameters:
                if p not in param_dict:
                    param_dict[p] = 0.0
            bound = self.circuit.bind_parameters(param_dict)
            compiled = transpile(bound, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            expectations = []
            for i in range(self.n_qubits):
                exp = 0.0
                for state, count in counts.items():
                    bit = state[-1-i]  # little‑endian
                    exp += (1 if bit == '0' else -1) * count
                exp /= self.shots
                expectations.append(exp)
            results.append(expectations)
        return np.array(results)

class HybridFunction(torch.autograd.Function):
    """Quantum‑to‑PyTorch bridge using the parameter‑shift rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        angles = inputs.detach().cpu().numpy()
        expectations = ctx.circuit.run(angles)
        out = torch.tensor(expectations, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        angles = inputs.detach().cpu().numpy()
        angles_plus = angles.copy()
        angles_minus = angles.copy()
        angles_plus[:, :] += shift
        angles_minus[:, :] -= shift
        exp_plus = ctx.circuit.run(angles_plus)
        exp_minus = ctx.circuit.run(angles_minus)
        grad = (exp_plus - exp_minus) / (2 * shift)
        grad_tensor = torch.tensor(grad, device=inputs.device, dtype=inputs.dtype)
        return grad_tensor * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

class QuantumHybridNAT(nn.Module):
    """CNN backbone followed by a quantum expectation head."""
    def __init__(self) -> None:
        super().__init__()
        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)
        backend = Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits=4, backend=backend, shots=200, shift=np.pi/2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.norm(x)
        x = self.hybrid(x)
        return x

__all__ = ["QuantumHybridNAT"]
