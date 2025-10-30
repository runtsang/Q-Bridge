"""Quantum implementation of SamplerQNNGen282 using Qiskit and a variational regression head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, random_unitary
import networkx as nx

# ------------------------------------------------------------------
# Fraud‑detection inspired parameter container
# ------------------------------------------------------------------
class FraudLayerParameters:
    def __init__(
        self,
        bs_theta: float,
        bs_phi: float,
        phases: tuple[float, float],
        squeeze_r: tuple[float, float],
        squeeze_phi: tuple[float, float],
        displacement_r: tuple[float, float],
        displacement_phi: tuple[float, float],
        kerr: tuple[float, float],
    ) -> None:
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

# ------------------------------------------------------------------
# Main quantum sampler
# ------------------------------------------------------------------
class SamplerQNNGen282:
    """
    Quantum sampler with:
        - a parameterized 2‑qubit sampler circuit (from SamplerQNN)
        - fraud‑detection inspired parameterized gates
        - a variational regression head for expectation value regression
    """

    def __init__(
        self,
        num_qubits: int,
        fraud_params: FraudLayerParameters,
        graph_threshold: float = 0.9,
    ) -> None:
        self.num_qubits = num_qubits
        self.fraud_params = fraud_params
        self.graph_threshold = graph_threshold
        self.backend = Aer.get_backend("statevector_simulator")
        self.circuit = self._build_circuit()

    def _sampler_circuit(
        self,
        inputs: ParameterVector,
        weights: ParameterVector,
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        return qc

    def _fraud_layer(self, qc: QuantumCircuit, params: FraudLayerParameters) -> None:
        # Simple mapping: use RY and RZ to mimic squeezing and displacement
        qc.ry(params.bs_theta, 0)
        qc.rz(params.bs_phi, 1)
        for phase in params.phases:
            qc.rz(phase, 0)
        for r, phi in zip(params.squeeze_r, params.squeeze_phi):
            qc.ry(r, 0)
            qc.rz(phi, 1)
        for r, phi in zip(params.displacement_r, params.displacement_phi):
            qc.rx(r, 0)
            qc.rz(phi, 1)
        for k in params.kerr:
            qc.rz(k, 0)

    def _regression_head(self, qc: QuantumCircuit) -> QuantumCircuit:
        # Measure all qubits in Z basis and return expectation values
        qc.measure_all()
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = self._sampler_circuit(inputs, weights)
        self._fraud_layer(qc, self.fraud_params)
        qc = self._regression_head(qc)
        return qc

    def sample(self, input_vals: np.ndarray, weight_vals: np.ndarray) -> np.ndarray:
        """Execute the circuit for given parameter values and return measurement probabilities."""
        param_dict = {"input": input_vals, "weight": weight_vals}
        bound_qc = self.circuit.bind_parameters(param_dict)
        job = execute(bound_qc, backend=self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts()
        probs = np.zeros(2 ** self.num_qubits)
        for bitstring, cnt in counts.items():
            probs[int(bitstring, 2)] = cnt / 1024
        return probs

    def compute_fidelity_adjacency(self, state_vectors: np.ndarray) -> nx.Graph:
        """
        Build a weighted graph from fidelity of state vectors.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(state_vectors.shape[0]))
        for i in range(state_vectors.shape[0]):
            for j in range(i + 1, state_vectors.shape[0]):
                fid = np.abs(np.vdot(state_vectors[i], state_vectors[j])) ** 2
                if fid >= self.graph_threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif fid >= self.graph_threshold * 0.8:
                    graph.add_edge(i, j, weight=0.5)
        return graph

    @staticmethod
    def random_network(num_qubits: int, layers: int) -> tuple[int, list[np.ndarray], np.ndarray]:
        """
        Generate a random unitary circuit (as a list of random unitary matrices) and a target unitary.
        """
        unitary_list = [random_unitary(2 ** n).data for n in range(1, layers + 1)]
        target = random_unitary(2 ** num_qubits).data
        return num_qubits, unitary_list, target

    @staticmethod
    def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Quantum regression data generation (same as ML side).
        """
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

__all__ = ["SamplerQNNGen282", "FraudLayerParameters"]
