"""Hybrid quantum sampler/classifier/regressor model."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
import torch
import torch.nn as nn

class QuantumSelfAttention:
    """Quantum implementation of a self‑attention block."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, name="self_attention_circuit")
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure_all()
        return qc

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        qc = self._build_circuit(rotation_params, entangle_params)
        job = execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)

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

class HybridSamplerModel:
    """Quantum hybrid model supporting sampling, classification and regression."""
    def __init__(self, mode: str, num_qubits: int, depth: int = 1, use_attention: bool = False, backend=None):
        self.mode = mode.lower()
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_attention = use_attention
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Parameter vectors
        self.input_params = ParameterVector("x", num_qubits)
        self.weight_params = ParameterVector("theta", num_qubits * depth)

        # Main circuit
        self.circuit = QuantumCircuit(num_qubits)
        for i, param in enumerate(self.input_params):
            self.circuit.rx(param, i)
        idx = 0
        for _ in range(depth):
            for i in range(num_qubits):
                self.circuit.ry(self.weight_params[idx], i)
                idx += 1
            for i in range(num_qubits - 1):
                self.circuit.cz(i, i + 1)

        # Add a simple random layer for regression
        if self.mode == "regressor":
            for i in range(num_qubits):
                self.circuit.ry(np.random.uniform(0, 2 * np.pi), i)

        # Attention circuit
        if use_attention:
            self.attention = QuantumSelfAttention(num_qubits)
        else:
            self.attention = None

        # Sampler for probabilities
        if self.mode in ("sampler", "classifier"):
            self.sampler = StatevectorSampler(self.backend)
            self.sampler_qnn = SamplerQNN(
                circuit=self.circuit,
                input_params=self.input_params,
                weight_params=self.weight_params,
                sampler=self.sampler,
            )
        else:
            self.sampler = None

        # Linear head for regression
        if self.mode == "regressor":
            self.head = nn.Linear(num_qubits, 1)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Execute the model on the given inputs."""
        if self.mode in ("sampler", "classifier"):
            param_dict = {str(p): val for p, val in zip(self.input_params, inputs)}
            state = self.sampler.sample(param_dict, shots=1024)
            probs = state.probabilities_dict()
            if self.mode == "sampler":
                return probs
            # For classifier, return logits for two basis states
            return np.array([probs.get("00", 0), probs.get("01", 0)])
        elif self.mode == "regressor":
            param_dict = {str(p): val for p, val in zip(self.input_params, inputs)}
            state = self.sampler.sample(param_dict, shots=1024)
            probs = state.probabilities_dict()
            expectations = []
            for i in range(self.num_qubits):
                exp = 0.0
                for bitstring, p in probs.items():
                    bit = int(bitstring[::-1][i])  # least significant qubit
                    exp += p * (-1) ** bit
                expectations.append(exp)
            features = np.array(expectations)
            return self.head(torch.from_numpy(features)).item()
        else:
            raise ValueError(f"Unsupported mode {self.mode}")

__all__ = ["HybridSamplerModel", "generate_superposition_data"]
