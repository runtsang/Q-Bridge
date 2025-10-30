import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit.random import random_circuit
from qiskit.circuit import Parameter
import torch
import torch.nn as nn
import torch.utils.data as data
from collections.abc import Iterable, Sequence
from typing import List

class QuantumHybridEstimator:
    """Hybrid estimator that evaluates parametrised quantum circuits for batches of inputs and observables,
    optionally adding Gaussian shot noise to emulate measurement statistics."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [complex(rng.normal(row[i].real, max(1e-6, 1 / shots)),
                                 rng.normal(row[i].imag, max(1e-6, 1 / shots))) for i in range(len(row))]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
#  Quantum convolution filter – adapted from pair 2
class QuantumConvFilter:
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, shots: int = 100):
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
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

# --------------------------------------------------------------------------- #
#  Quantum self‑attention – adapted from pair 4
class QuantumSelfAttention:
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

# --------------------------------------------------------------------------- #
#  Quantum regression dataset – adapted from pair 3
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
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
    return states, labels

class QuantumRegressionDataset(data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
#  Quantum hybrid model – combines encoding, a variational layer, measurement and a classical head
class QuantumHybridModel(nn.Module):
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = [qiskit.circuit.library.RY(0.0) for _ in range(num_wires)]
        self.var_layer = random_circuit(num_wires, 2)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        outputs = []
        for i in range(bsz):
            qdev = Statevector(state_batch[i].numpy())
            for w in range(self.num_wires):
                qdev = qdev.evolve(qiskit.circuit.library.RY(state_batch[i][w].item()))
            qdev = qdev.evolve(self.var_layer)
            exp_vals = [qdev.expectation_value(qiskit.circuit.library.PauliZ()) for _ in range(self.num_wires)]
            outputs.append(exp_vals)
        features = torch.tensor(outputs, dtype=torch.float32, device=state_batch.device)
        return self.head(features).squeeze(-1)

__all__ = [
    "QuantumHybridEstimator",
    "QuantumConvFilter",
    "QuantumSelfAttention",
    "QuantumRegressionDataset",
    "generate_superposition_data",
    "QuantumHybridModel",
]
