import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator

# Photonic‑style layer parameters and helper functions
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    def __init__(self, bs_theta, bs_phi, phases, squeeze_r, squeeze_phi,
                 displacement_r, displacement_phi, kerr):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes, params: FraudLayerParameters, *, clip: bool) -> None:
    QuantumCircuit(bs_theta=params.bs_theta, bs_phi=params.bs_phi).to_gate()(modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        QuantumCircuit(rgate=phase).to_gate()(modes[i])
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        QuantumCircuit(sgate=r if not clip else _clip(r, 5), phi=phi).to_gate()(modes[i])
    QuantumCircuit(bs_theta=params.bs_theta, bs_phi=params.bs_phi).to_gate()(modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        QuantumCircuit(rgate=phase).to_gate()(modes[i])
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        QuantumCircuit(dgate=r if not clip else _clip(r, 5), phi=phi).to_gate()(modes[i])
    for i, k in enumerate(params.kerr):
        QuantumCircuit(kgate=k if not clip else _clip(k, 1)).to_gate()(modes[i])

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a classical sequential model that mimics the photonic structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# Quantum classifier circuit factory
def build_classifier_circuit(num_qubits: int, depth: int):
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables

class QuantumCircuitWrapper:
    def __init__(self, circuit: QuantumCircuit, backend, shots: int):
        self.circuit = circuit
        self.backend = backend
        self.shots = shots
        self.compiled = transpile(self.circuit, self.backend)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        param_binds = [{self.circuit.parameters[i]: val for i, val in enumerate(thetas)}]
        qobj = assemble(self.compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        expectation = ctx.circuit.run(inputs.tolist())
        return torch.tensor(expectation, dtype=inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs = ctx.circuit
        shift = ctx.shift
        grads = []
        for val in inputs.tolist():
            right = ctx.circuit.run([val + shift])
            left = ctx.circuit.run([val - shift])
            grads.append(right - left)
        grad = torch.tensor(grads, dtype=grad_output.dtype)
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2) -> None:
        super().__init__()
        circuit, _, _, _ = build_classifier_circuit(n_qubits, depth=2)
        self.quantum = QuantumCircuitWrapper(circuit, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.quantum, self.shift)

class FraudDetectionModel(nn.Module):
    """Hybrid fraud‑detection model that uses a photonic base network and a
    Qiskit‑based quantum expectation head."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers_params: Iterable[FraudLayerParameters],
        n_qubits: int = 2,
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.base = build_fraud_detection_program(input_params, layers_params)
        backend = AerSimulator()
        self.head = Hybrid(n_qubits, backend, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        x = self.head(x)
        return torch.cat((x, 1 - x), dim=-1)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "QuantumCircuitWrapper",
    "HybridFunction",
    "Hybrid",
    "FraudDetectionModel",
]
