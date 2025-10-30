"""Hybrid quantum kernel with autoencoder and estimator."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as _EstimatorQNN

class QuantumAutoencoder:
    """Quantum autoencoder using a RealAmplitudes ansatz and a swap‑test."""
    def __init__(self, num_latent: int, num_trash: int) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.num_qubits = num_latent + 2 * num_trash + 1
        total_params = num_latent + num_trash
        self.param_list = [Parameter(f"p_{i}") for i in range(total_params)]
        self.base_circuit = self._build_base_circuit()

    def _build_base_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=5)
        qc.compose(ansatz.bind_parameters(self.param_list), range(0, self.num_latent + self.num_trash), inplace=True)
        qc.barrier()
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def get_circuit(self, param_values: list[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.param_list):
            raise ValueError("Parameter list length mismatch.")
        bound_circuit = self.base_circuit.bind_parameters(dict(zip(self.param_list, param_values)))
        return bound_circuit


class QuantumKernel:
    """Swap‑test kernel for two real vectors."""
    def __init__(self, num_qubits: int = 4) -> None:
        self.num_qubits = num_qubits
        self.sampler = StatevectorSampler()

    def __call__(self, x: list[float], y: list[float]) -> float:
        qc = self._build_swap_test_circuit(x, y)
        result = self.sampler.run(qc).result()
        state = result.get_statevector()
        return abs(state[0]) ** 2

    def _build_swap_test_circuit(self, x: list[float], y: list[float]) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits * 2, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        for i, val in enumerate(x):
            qc.ry(val, i)
        for i, val in enumerate(y):
            qc.ry(val, i + self.num_qubits)
        qc.h(self.num_qubits)
        for i in range(self.num_qubits):
            qc.cswap(self.num_qubits, i, i + self.num_qubits)
        qc.h(self.num_qubits)
        qc.measure(self.num_qubits, cr[0])
        return qc


class HybridKernel:
    """Hybrid kernel that compresses inputs via a quantum autoencoder and then applies a swap‑test kernel."""
    def __init__(self, num_qubits: int = 4, latent_dim: int = 3, num_trash: int = 2) -> None:
        self.autoencoder = QuantumAutoencoder(latent_dim, num_trash)
        self.kernel = QuantumKernel(num_qubits=latent_dim + num_trash)

    def __call__(self, x: list[float], y: list[float]) -> float:
        qc = self._build_hybrid_circuit(x, y)
        sampler = StatevectorSampler()
        result = sampler.run(qc).result()
        state = result.get_statevector()
        return abs(state[0]) ** 2

    def _build_hybrid_circuit(self, x: list[float], y: list[float]) -> QuantumCircuit:
        n = self.autoencoder.num_qubits
        total_qubits = 2 * n
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.append(self.autoencoder.base_circuit, range(0, n))
        qc = qc.bind_parameters(dict(zip(self.autoencoder.param_list, x)))
        qc.append(self.autoencoder.base_circuit, range(n, 2 * n))
        qc = qc.bind_parameters(dict(zip(self.autoencoder.param_list, y)))
        qc.h(n)
        for i in range(n):
            qc.cswap(n, i, i + n)
        qc.h(n)
        qc.measure(n, cr[0])
        return qc


class EstimatorQNN:
    """Quantum neural network regressor."""
    def __init__(self) -> None:
        self.params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(self.params[0], 0)
        qc.rx(self.params[1], 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        self.qnn = _EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[self.params[0]],
            weight_params=[self.params[1]],
            estimator=estimator,
        )

    def predict(self, input_vals: list[float]) -> float:
        return self.qnn.predict(input_vals)[0]


def kernel_matrix(a: list[list[float]], b: list[list[float]]) -> np.ndarray:
    """Compute Gram matrix between two lists of real vectors using HybridKernel."""
    kernel = HybridKernel()
    return np.array([[kernel(x, y) for y in b] for x in a])


__all__ = [
    "QuantumAutoencoder",
    "QuantumKernel",
    "HybridKernel",
    "EstimatorQNN",
    "kernel_matrix",
]
