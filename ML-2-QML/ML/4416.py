import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# --------------------------------------------------------------------------- #
#  Quantum circuit wrapper – parameterised ansatz with Ry‑CZ layers
# --------------------------------------------------------------------------- #
class QuantumCircuitWrapper:
    """
    Builds a layered variational circuit with explicit encoding via RX and
    depth‑dependent Ry + CZ interactions.  The circuit is executed on the
    Aer simulator and returns the Z‑expectation of each qubit.
    """
    def __init__(self, n_qubits: int, depth: int) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self._build_circuit()
        self.backend = Aer.get_backend("aer_simulator_statevector")
        self.shots = 1024

    def _build_circuit(self) -> None:
        self.circuit = QuantumCircuit(self.n_qubits)
        # Encode data with RX
        encoding = ParameterVector("x", self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.rx(encoding[i], i)
        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for i in range(self.n_qubits):
                self.circuit.ry(ParameterVector(f"theta_{idx}", 1)[0], i)
                idx += 1
            for i in range(self.n_qubits - 1):
                self.circuit.cz(i, i + 1)

    def run(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of parameter vectors.
        Each row in `params` corresponds to one sample.
        Returns a 2‑D array of shape (batch, n_qubits) with Z‑expectations.
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)
        n_samples = params.shape[0]
        expectations = np.zeros((n_samples, self.n_qubits))
        for i in range(n_samples):
            param_bind = {
                f"x_{j}": params[i, j] for j in range(self.n_qubits)
            }
            # Map variational parameters
            var_params = params[i, self.n_qubits:]
            for k, val in enumerate(var_params):
                param_bind[f"theta_{k}"] = val
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[param_bind])
            job = self.backend.run(qobj)
            result = job.result()
            statevector = result.get_statevector(compiled)
            # Compute expectation of Z for each qubit
            for q in range(self.n_qubits):
                exp = 0.0
                for idx, amp in enumerate(statevector):
                    bit = (idx >> q) & 1
                    exp += ((-1) ** bit) * (np.abs(amp) ** 2)
                expectations[i, q] = exp
        return expectations

# --------------------------------------------------------------------------- #
#  Differentiable interface using finite‑difference gradients
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """
    Wraps the Qiskit circuit to make it differentiable via central finite
    differences.  The shift value is a small constant controlling the
    approximation quality.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float = 1e-3) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert to numpy for Qiskit
        params = inputs.detach().cpu().numpy()
        expectations = circuit.run(params)
        # We use the expectation of the first qubit as the scalar output
        result = torch.tensor(expectations[:, 0], device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        params = inputs.detach().cpu().numpy()
        grad = np.zeros_like(params)
        for i in range(params.shape[0]):
            for j in range(params.shape[1]):
                perturbed_plus = params.copy()
                perturbed_minus = params.copy()
                perturbed_plus[i, j] += shift
                perturbed_minus[i, j] -= shift
                exp_plus = circuit.run(perturbed_plus)[i, 0]
                exp_minus = circuit.run(perturbed_minus)[i, 0]
                grad[i, j] = (exp_plus - exp_minus) / (2 * shift)
        grad_tensor = torch.tensor(grad, device=inputs.device, dtype=inputs.dtype)
        return grad_tensor * grad_output.unsqueeze(-1), None, None

# --------------------------------------------------------------------------- #
#  Hybrid head – a single‑qubit expectation layer
# --------------------------------------------------------------------------- #
class HybridHead(nn.Module):
    def __init__(self, n_qubits: int = 1, depth: int = 2, shift: float = 1e-3) -> None:
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, depth)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, n_qubits)
        return HybridFunction.apply(inputs, self.circuit, self.shift)

# --------------------------------------------------------------------------- #
#  Classical‑Quantum hybrid classifier
# --------------------------------------------------------------------------- #
class HybridBinaryClassifier(nn.Module):
    """
    CNN + FC backbone followed by a quantum expectation head.
    The model can be switched to a fully classical head via the
    `build_classical_equivalent` factory.
    """
    def __init__(self, n_qubits: int = 1, depth: int = 2) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Quantum head
        self.hybrid_head = HybridHead(n_qubits, depth)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
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
        # Quantum expectation
        prob = torch.sigmoid(self.hybrid_head(x))
        return torch.cat((prob, 1 - prob), dim=-1)

    @staticmethod
    def build_classical_equivalent(num_features: int, depth: int):
        """
        Factory that returns a purely classical feed‑forward network mirroring
        the structure of the quantum head.  The implementation follows
        the pattern used in `build_classifier_circuit` from the reference
        pair.
        """
        layers = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
        head = nn.Linear(num_features, 2)
        layers.append(head)
        return nn.Sequential(*layers)

# --------------------------------------------------------------------------- #
#  Synthetic dataset generator for superposition‑based binary labels
# --------------------------------------------------------------------------- #
def generate_binary_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a dataset where labels are determined by the sign of
    sin(sum(x)), creating a non‑linear decision boundary that is
    challenging for classical models and well‑suited for quantum
    expectation heads.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = (np.sin(angles) > 0).astype(np.float32)
    return x, y

__all__ = [
    "HybridFunction",
    "HybridHead",
    "HybridBinaryClassifier",
    "generate_binary_superposition_data",
]
