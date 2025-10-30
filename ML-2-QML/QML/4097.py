import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
import torchquantum as tq
from typing import Iterable, Sequence, Union, List

class QuantumSelfAttention:
    """
    Quantum self‑attention circuit that mirrors the classical block.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # rotation_params shape: (3*n_qubits,)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # entangle_params shape: (n_qubits-1,)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

class FastBaseEstimator:
    """
    Estimator that can evaluate either a Qiskit QuantumCircuit or a torchquantum QuantumModule.
    """
    def __init__(self, model: Union[QuantumCircuit, tq.QuantumModule]):
        self.model = model

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int = 1024,
        backend: qiskit.providers.Backend | None = None,
    ) -> List[List[complex]]:
        if isinstance(self.model, QuantumCircuit):
            return self._evaluate_circuit(observables, parameter_sets, shots, backend)
        elif isinstance(self.model, tq.QuantumModule):
            return self._evaluate_quantum_module(observables, parameter_sets, shots)
        else:
            raise TypeError("Unsupported model type for FastBaseEstimator")

    def _evaluate_circuit(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        backend: qiskit.providers.Backend | None,
    ) -> List[List[complex]]:
        backend = backend or qiskit.Aer.get_backend("aer_simulator_statevector")
        results: List[List[complex]] = []
        for params in parameter_sets:
            circ = self.model.assign_parameters(dict(zip(self.model.parameters, params)), inplace=False)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _evaluate_quantum_module(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
    ) -> List[List[complex]]:
        # For torchquantum modules we assume observables are PauliZ on individual wires.
        results: List[List[complex]] = []
        for params in parameter_sets:
            qdev = tq.QuantumDevice(n_wires=self.model.n_wires, bsz=1, device="cpu")
            self.model(qdev)
            meas = tq.MeasureAll(tq.PauliZ)
            features = meas(qdev)
            row = [complex(f.item()) for f in features]
            results.append(row)
        return results

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

class RegressionDataset(torch.utils.data.Dataset):
    """
    Quantum regression dataset generating superposition states.
    """
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
    """
    Quantum neural network that mirrors the classical QModel.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

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

__all__ = ["FastBaseEstimator", "QuantumSelfAttention", "RegressionDataset", "QModel", "generate_superposition_data"]
