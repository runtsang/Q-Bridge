"""Quantum estimator with autoencoder, hybrid classification, and regression support."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit as QC, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit import assemble, transpile
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

# --------------------------------------------------------------------------- #
#  Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """Wrapper around a parametrised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = QC(n_qubits)
        self._circuit.h(list(range(n_qubits)))
        self._circuit.barrier()
        self.theta = Parameter("theta")
        self._circuit.ry(self.theta, list(range(n_qubits)))
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

# --------------------------------------------------------------------------- #
#  Hybrid layer
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
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
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)
        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

# --------------------------------------------------------------------------- #
#  Hybrid classifier
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
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(self.fc3.out_features, backend, shots=100, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = nn.functional.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.drop2(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)

# --------------------------------------------------------------------------- #
#  Quantum autoencoder
# --------------------------------------------------------------------------- #
def Autoencoder(num_latent: int, num_trash: int) -> SamplerQNN:
    """Build a quantum autoencoder circuit and return a SamplerQNN."""
    def ansatz(num_qubits):
        return RealAmplitudes(num_qubits, reps=5)

    def auto_encoder_circuit(num_latent, num_trash):
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QC(qr, cr)
        circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()
        aux = num_latent + 2 * num_trash
        circuit.h(aux)
        for i in range(num_trash):
            circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    circuit = auto_encoder_circuit(num_latent, num_trash)
    sampler = StatevectorSampler()
    return SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )

# --------------------------------------------------------------------------- #
#  Regression dataset
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
#  Unified quantum estimator
# --------------------------------------------------------------------------- #
class FastHybridEstimator:
    """Unified estimator that evaluates quantum circuits with optional shot‑noise simulation."""
    def __init__(
        self,
        circuit: QC,
        *,
        shots: int | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.shots = shots
        self.rng = np.random.default_rng(rng_seed)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables) or [Statevector]
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound_circ = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, params)), inplace=False)
            if self.shots is None:
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # use StatevectorSampler for noisy expectation
                sampler = StatevectorSampler()
                counts = sampler.run(bound_circ, shots=self.shots).get_counts(bound_circ)
                def expectation(counts_dict):
                    probs = np.array(list(counts_dict.values())) / self.shots
                    states = np.array([int(k, 2) for k in counts_dict.keys()])
                    return np.sum(states * probs)
                row = [expectation(counts) for obs in observables]
            if self.shots is not None:
                noisy_row = [
                    float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row
                ]
                row = noisy_row
            results.append(row)
        return results

__all__ = [
    "FastHybridEstimator",
    "Autoencoder",
    "RegressionDataset",
    "generate_superposition_data",
    "QModel",
    "QuantumCircuit",
    "HybridFunction",
    "Hybrid",
    "QCNet",
]
