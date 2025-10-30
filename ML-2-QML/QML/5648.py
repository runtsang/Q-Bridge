"""
Hybrid quantum attention module that mirrors the classical hybrid design.
It builds a parameterised self‑attention circuit and pairs it with a
SamplerQNN to generate probability distributions that can be used to
weight the attention outputs.

The module uses Qiskit and the Qiskit Machine Learning package to
construct a two‑qubit sampler and a four‑qubit attention circuit.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.compiler import transpile
from qiskit.providers import Backend
from qiskit_aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class HybridQuantumAttention:
    """
    Quantum self‑attention block coupled with a SamplerQNN.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the attention circuit (default 4).
    backend : Backend | None
        Qiskit backend to execute circuits. Defaults to Aer qasm simulator.
    shots : int
        Number of shots for measurement.
    """

    def __init__(self, n_qubits: int = 4, backend: Backend | None = None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots

        # Parameter vectors for the attention circuit
        self.rotation_params = ParameterVector("rot", 3 * n_qubits)
        self.entangle_params = ParameterVector("ent", n_qubits - 1)

        # Build the sampler circuit (two‑qubit example)
        self.sampler_circuit = self._build_sampler_circuit()
        self.sampler_qnn = SamplerQNN(
            circuit=self.sampler_circuit,
            input_params=ParameterVector("in", 2),
            weight_params=self.rotation_params,
            sampler=StatevectorSampler()
        )

    def _build_sampler_circuit(self) -> QuantumCircuit:
        """Constructs a simple parameterised sampler circuit."""
        qc = QuantumCircuit(2)
        qc.ry(ParameterVector("in", 2)[0], 0)
        qc.ry(ParameterVector("in", 2)[1], 1)
        qc.cx(0, 1)
        qc.ry(self.rotation_params[0], 0)
        qc.ry(self.rotation_params[1], 1)
        qc.cx(0, 1)
        qc.ry(self.rotation_params[2], 0)
        qc.ry(self.rotation_params[3], 1)
        return qc

    def _build_attention_circuit(self) -> QuantumCircuit:
        """Builds the 4‑qubit attention circuit."""
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(self.rotation_params[3 * i], i)
            qc.ry(self.rotation_params[3 * i + 1], i)
            qc.rz(self.rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(self.entangle_params[i], i, i + 1)
        qc.measure_all()
        return qc

    def run(self, input_values: np.ndarray, param_dict: Dict[str, float] | None = None) -> Dict[str, Any]:
        """
        Execute the hybrid attention circuit.

        Parameters
        ----------
        input_values : np.ndarray
            Array of shape (2,) representing the input parameters for the sampler.
        param_dict : Dict[str, float] | None
            Mapping from parameter names to values. If None, random values are used.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the sampler probabilities and the
            attention measurement counts.
        """
        # Bind sampler parameters
        sampler_params = {p: val for p, val in zip(self.sampler_circuit.parameters, input_values)}
        sampler_counts = self.sampler_qnn.run(input_values, sampler_params)

        # Generate random or supplied parameters for the attention circuit
        if param_dict is None:
            param_dict = {p: np.random.rand() for p in self._build_attention_circuit().parameters}

        # Build and execute attention circuit
        attn_circuit = self._build_attention_circuit()
        attn_circuit.bind_parameters(param_dict)
        transpiled = transpile(attn_circuit, self.backend)
        job = self.backend.run(transpiled, shots=self.shots)
        attn_counts = job.result().get_counts(transpiled)

        return {"sampler_counts": sampler_counts, "attention_counts": attn_counts}

__all__ = ["HybridQuantumAttention"]
