"""Quantum‑centric implementation of a hybrid binary classifier that mirrors
the classical structure but replaces the final head with a parameterised quantum
circuit.  The module also provides a QCNN quantum circuit and a lightweight
estimator for expectation values."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap

# --------------------------------------------------------------------------- #
# Parametric circuit wrapper
# --------------------------------------------------------------------------- #
class ParamCircuit:
    """Simple two‑qubit circuit with a single parameter."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.circuit = QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = QuantumCircuit.Parameter("theta")
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
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
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------------- #
# Hybrid head that forwards activations through the parametric circuit
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: ParamCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z], dtype=torch.float32)
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
        gradients = torch.tensor([gradients], dtype=torch.float32)
        return gradients * grad_output, None, None

class Hybrid(nn.Module):
    """Layer that forwards activations through a parameterised quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = ParamCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

# --------------------------------------------------------------------------- #
# Quantum‑enhanced CNN backbone
# --------------------------------------------------------------------------- #
class QCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = AerSimulator()
        self.hybrid = Hybrid(
            n_qubits=self.fc3.out_features,
            backend=backend,
            shots=100,
            shift=np.pi / 2,
        )

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
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)

# --------------------------------------------------------------------------- #
# QCNN quantum circuit (simplified)
# --------------------------------------------------------------------------- #
def QCNNQuantum() -> QuantumCircuit:
    """Simplified QCNN ansatz: 8‑qubit feature map followed by a parametric layer."""
    feature_map = ZFeatureMap(8)
    params = ParameterVector("theta", length=8)
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, qubits=range(8), inplace=True)
    for i in range(8):
        circuit.ry(params[i], i)
    circuit.measure_all()
    return circuit

# --------------------------------------------------------------------------- #
# Fast estimator for quantum circuits
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[SparsePauliOp], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
# Unified quantum wrapper
# --------------------------------------------------------------------------- #
class HybridBinaryClassifier:
    """Quantum‑centric binary classifier that mirrors the classical API."""
    def __init__(self, use_qcnn: bool = False, shift: float = 0.0) -> None:
        self.backbone = QCNet()
        self.use_qcnn = use_qcnn
        if use_qcnn:
            self.feature_extractor = QCNNQuantum()
        else:
            self.feature_extractor = None
        self.hybrid_head = Hybrid(
            n_qubits=1,
            backend=AerSimulator(),
            shots=100,
            shift=shift,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.backbone(inputs)
        if self.use_qcnn:
            # Evaluate QCNN features (placeholder: use a simple estimator)
            estimator = FastBaseEstimator(self.feature_extractor)
            # Convert backbone output to parameters for QCNN (pad to 8 qubits)
            params = torch.cat([x, torch.zeros(8 - x.shape[1], device=x.device)], dim=1)
            qc_features = torch.tensor(
                estimator.evaluate([SparsePauliOp.from_list([("Z", 1)])], params.tolist()),
                dtype=torch.float32,
            )
            x = torch.cat([x, qc_features], dim=-1)
        return self.hybrid_head(x)

__all__ = [
    "ParamCircuit",
    "HybridFunction",
    "Hybrid",
    "QCNet",
    "QCNNQuantum",
    "FastBaseEstimator",
    "HybridBinaryClassifier",
]
