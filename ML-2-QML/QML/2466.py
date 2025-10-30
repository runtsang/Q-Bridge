"""Quantum hybrid estimator combining a quanvolution circuit with a variational EstimatorQNN."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
import torch


class QuanvCircuit:
    """Quantum convolution filter used in quanvolution layers."""
    def __init__(self, kernel_size: int = 2,
                 backend: qiskit.providers.Backend | None = None,
                 shots: int = 100,
                 threshold: float = 127.0) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the quantum circuit on a single 2‑D array of shape (kernel_size, kernel_size)."""
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0
                    for i, val in enumerate(dat)}
            param_binds.append(bind)

        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


class HybridEstimator:
    """Hybrid quantum neural network that first extracts features with a QuanvCircuit
    and then estimates a scalar target with a variational EstimatorQNN."""
    def __init__(self, kernel_size: int = 2,
                 backend: qiskit.providers.Backend | None = None,
                 shots: int = 100,
                 threshold: float = 127.0,
                 weight_init: float = 0.1) -> None:
        self.feature_extractor = QuanvCircuit(kernel_size, backend, shots, threshold)

        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)

        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=SparsePauliOp.from_list([("Y", 1)]),
            input_params=[params[0]],
            weight_params=[params[1]],
            estimator=Estimator(),
        )

        init_val = torch.tensor(weight_init, dtype=torch.float64)
        self.estimator_qnn.set_weights(torch.tensor([init_val], dtype=torch.float64))

    def run(self, data: np.ndarray) -> float:
        """
        Args:
            data: 2‑D array of shape (kernel_size, kernel_size)
        Returns:
            float: predicted scalar value from the hybrid QNN.
        """
        feature = self.feature_extractor.run(data)
        return float(self.estimator_qnn.predict(np.array([feature])))


def get_hybrid_qnn(kernel_size: int = 2,
                   backend: qiskit.providers.Backend | None = None,
                   shots: int = 100,
                   threshold: float = 127.0,
                   weight_init: float = 0.1) -> HybridEstimator:
    """Convenience factory returning a ready‑to‑use HybridEstimator."""
    return HybridEstimator(kernel_size, backend, shots, threshold, weight_init)


__all__ = ["QuanvCircuit", "HybridEstimator", "get_hybrid_qnn"]
