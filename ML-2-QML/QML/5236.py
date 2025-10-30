import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZFeatureMap


class QCNNHybrid(tq.QuantumModule):
    """
    Quantum implementation of the QCNN architecture.  It stitches together
    a feature map, a stack of convolution‑ and pooling‑like two‑qubit blocks,
    and a single‑qubit measurement head.  The ansatz mirrors the classical
    stages but uses parametric gates so that gradient‑based optimisation
    can be performed with a qiskit estimator.
    """
    def __init__(self, num_qubits: int = 8):
        super().__init__()
        self.num_qubits = num_qubits
        self.feature_map = ZFeatureMap(num_qubits)
        self.ansatz = self._build_ansatz()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self):
        qc = QuantumCircuit(self.num_qubits)
        qc.append(self._conv_layer(8, "c1"), range(8))
        qc.append(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8))
        qc.append(self._conv_layer(4, "c2"), range(4, 8))
        qc.append(self._pool_layer([0, 1], [2, 3], "p2"), range(4, 8))
        qc.append(self._conv_layer(2, "c3"), range(6, 8))
        qc.append(self._pool_layer([0], [1], "p3"), range(6, 8))
        return qc

    def _conv_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _conv_layer(self, num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            qc.append(self._conv_circuit(param_vec[idx : idx + 3]), [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def _pool_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources, sinks, prefix):
        num = len(sources) + len(sinks)
        qc = QuantumCircuit(num)
        param_vec = ParameterVector(prefix, length=num // 2 * 3)
        idx = 0
        for s, t in zip(sources, sinks):
            qc.append(self._pool_circuit(param_vec[idx : idx + 3]), [s, t])
            qc.barrier()
            idx += 3
        return qc

    def forward(self, states):
        """
        Forward pass takes a batch of classical feature vectors (shape
        [batch, num_qubits]) and returns the QCNN output.
        """
        return self.qnn(states)


class RegressionDataset(torch.utils.data.Dataset):
    """
    Quantum dataset of superposition states and regression targets.
    Data is generated as |ψ⟩ = cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩.
    """
    def __init__(self, samples: int = 1000, num_wires: int = 8):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def generate_superposition_data(num_wires: int, samples: int):
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


class FastBaseEstimator:
    """
    Minimal estimator that evaluates a parametrised circuit for a list of
    parameter sets and a list of observables.  It mimics the behaviour of
    the classical FastEstimator but operates on Statevectors.
    """
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values):
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables, parameter_sets):
        observables = list(observables)
        results = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


__all__ = ["QCNNHybrid", "RegressionDataset", "generate_superposition_data", "FastBaseEstimator"]
