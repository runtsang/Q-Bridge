import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.circuit.random import random_circuit

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantum analogue of generate_superposition_data."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumCircuit:
    """Parametrised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
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
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.circuit.run([value + shift[idx]])
            expectation_left = ctx.circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)
        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

class QuanvCircuit:
    """Quantum filter circuit used as a quanvolution layer."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class EstimatorQNN(nn.Module):
    """Quantum estimator based on Qiskit EstimatorQNN."""
    def __init__(self, num_wires: int):
        super().__init__()
        params1 = [Parameter("input1"), Parameter("weight1")]
        qc1 = qiskit.QuantumCircuit(1)
        qc1.h(0)
        qc1.ry(params1[0], 0)
        qc1.rx(params1[1], 0)
        observable1 = SparsePauliOp.from_list([("Y", 1)])
        estimator = Estimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=qc1,
            observables=observable1,
            input_params=[params1[0]],
            weight_params=[params1[1]],
            estimator=estimator,
        )
        self.num_wires = num_wires

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = []
        for inp in inputs:
            out = self.estimator_qnn.predict(np.array([inp.item()]))
            outputs.append(out[0])
        return torch.tensor(outputs, dtype=torch.float32)

class HybridRegressionModel(nn.Module):
    """Hybrid quantum‑classical regression model that fuses a quanvolution layer,
    dense feature extractor and a quantum hybrid head."""
    def __init__(self, num_features: int):
        super().__init__()
        # Quanvolution filter
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.quanv = QuanvCircuit(kernel_size=2, backend=backend, shots=100, threshold=0.5)
        # Dense feature extractor
        self.fc = nn.Sequential(
            nn.Linear(num_features + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Hybrid quantum head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits=1, backend=backend, shots=200, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, features)
        size = int(np.sqrt(inputs.shape[1]))
        # reshape for quanvolution
        quanv_input = inputs.view(inputs.shape[0], 1, size, size)
        # run quanvolution filter on each patch
        conv_feat = []
        for patch in quanv_input.unbind(1):
            patch_flat = patch.view(-1).numpy()
            conv_feat.append(self.quanv.run(patch_flat))
        conv_feat = torch.tensor(conv_feat, dtype=torch.float32).unsqueeze(-1)
        # concatenate with raw features
        x = torch.cat([conv_feat, inputs], dim=1)
        x = self.fc(x)
        return self.hybrid(x)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data", "EstimatorQNN"]
