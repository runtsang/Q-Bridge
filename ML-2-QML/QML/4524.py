import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
import numpy as np
from typing import Iterable, List, Tuple

# ----------------------------------------------------------------------
#  Quantum circuit builder (from reference 3)
# ----------------------------------------------------------------------
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[qiskit.QuantumCircuit,
                                                                   List[qiskit.circuit.Parameter],
                                                                   List[qiskit.circuit.Parameter],
                                                                   List[qiskit.quantum_info.SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    encoding = qiskit.circuit.ParameterVector("x", num_qubits)
    weights = qiskit.circuit.ParameterVector("theta", num_qubits * depth)

    circuit = qiskit.QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [qiskit.quantum_info.SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# ----------------------------------------------------------------------
#  Hybrid function (classical‑quantum interface)
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper that forwards a scalar through a quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: qiskit.QuantumCircuit,
                shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit

        # Run the circuit with the current parameters
        compiled = transpile(circuit, ctx.circuit.backend)
        job = ctx.circuit.backend.run(assemble(compiled, shots=ctx.circuit.shots))
        result = job.result()
        counts = result.get_counts()
        exp = 0.0
        for bitstring, count in counts.items():
            # Expectation of Z on the first qubit
            z = 1 if bitstring[-1] == '0' else -1
            exp += z * count
        exp /= ctx.circuit.shots
        output = torch.tensor([exp], dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs) * ctx.shift
        gradients = []
        for idx, value in enumerate(inputs.tolist()):
            # Parameter‑shift rule
            right = ctx.circuit.run([value + shift[idx]])
            left = ctx.circuit.run([value - shift[idx]])
            gradients.append(right - left)
        gradients = torch.tensor(gradients, dtype=grad_output.dtype, device=grad_output.device)
        return gradients * grad_output, None, None

# ----------------------------------------------------------------------
#  Quantum hybrid head
# ----------------------------------------------------------------------
class QuantumHybridHead(nn.Module):
    """Variational quantum circuit that outputs a single expectation value."""

    def __init__(self, num_qubits: int, depth: int, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        circuit, encoding, weights, _ = build_classifier_circuit(num_qubits, depth)
        self.circuit = circuit
        self.circuit.backend = self.backend
        self.circuit.shots = shots
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Flatten input to match circuit parameters
        flat = torch.flatten(inputs, 1)
        # Bind parameters to the circuit
        param_dict = {}
        for i, val in enumerate(flat[0, :self.circuit.num_qubits]):
            param_dict[self.circuit.parameters[i]] = val.item()
        # Variational parameters set to zero
        for j in range(self.circuit.num_qubits * self.circuit.num_qubits):
            param_dict[self.circuit.parameters[self.circuit.num_qubits + j]] = 0.0
        bound = self.circuit.bind_parameters(param_dict)
        return HybridFunction.apply(bound, self.circuit, self.shift)

# ----------------------------------------------------------------------
#  Quantum kernel (simple swap‑test variant)
# ----------------------------------------------------------------------
class QuantumKernel(nn.Module):
    """Compute a similarity score between two vectors using a 1‑qubit swap‑test style circuit."""

    def __init__(self, num_qubits: int, backend=None, shots: int = 1024):
        super().__init__()
        self.num_qubits = num_qubits
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots

    def _build_circuit(self, x: np.ndarray, y: np.ndarray) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            qc.ry(x[i], i)
            qc.ry(-y[i], i)
        qc.measure_all()
        return qc

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        kernels = []
        for xi, yi in zip(x_np, y_np):
            qc = self._build_circuit(xi, yi)
            compiled = transpile(qc, self.backend)
            job = self.backend.run(assemble(compiled, shots=self.shots))
            result = job.result()
            counts = result.get_counts()
            exp = 0.0
            for bitstring, count in counts.items():
                z = 1 if bitstring[-1] == '0' else -1
                exp += z * count
            exp /= self.shots
            kernels.append(exp)
        return torch.tensor(kernels, dtype=torch.float32)

# ----------------------------------------------------------------------
#  Quantum hybrid classifier (end‑to‑end)
# ----------------------------------------------------------------------
class QuantumHybridClassifier(nn.Module):
    """
    CNN backbone followed by a variational quantum circuit.
    The quantum head outputs a single expectation value that is passed
    through a sigmoid to obtain class probabilities.
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_qubits: int = 5,
                 depth: int = 2,
                 backend=None,
                 shots: int = 1024):
        super().__init__()
        # CNN backbone (same as classical)
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum head
        self.quantum_head = QuantumHybridHead(num_qubits, depth,
                                              backend=backend,
                                              shots=shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extractor
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

        # Quantum expectation head
        q_out = self.quantum_head(x)
        probs = torch.sigmoid(q_out)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumHybridClassifier", "QuantumKernel", "QuantumHybridHead"]
