"""
HybridSamplerAttentionQuantum – quantum implementation of sampler + attention.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.providers import Backend
from qiskit_aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler


class HybridSamplerAttentionQuantum:
    """
    Quantum hybrid network that interleaves a parameterised sampler
    circuit with a self‑attention style block.  All parameters are
    grouped into input, weight, rotation, and entangle vectors.
    """

    def __init__(self, n_qubits: int = 4) -> None:
        """
        Parameters
        ----------
        n_qubits: int
            Number of qubits; must be even so that half are treated as
            sampler inputs and half as attention registers.
        """
        self.n_qubits = n_qubits
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)
        self.rot_params = ParameterVector("rot", 3 * n_qubits)
        self.ent_params = ParameterVector("ent", n_qubits - 1)

        # Backend for execution
        self.backend: Backend = AerSimulator()

    def _build_sampler_circuit(self) -> QuantumCircuit:
        """
        Build the sampler part of the circuit.  Mirrors the classical
        linear network but implemented with Ry rotations and CX entanglement.
        """
        qc = QuantumCircuit(self.n_qubits)
        # Apply input rotations
        for i in range(2):
            qc.ry(self.input_params[i], i)
        # Entangle first two qubits
        qc.cx(0, 1)
        # Apply weight rotations
        for i in range(4):
            qc.ry(self.weight_params[i], i)
        # Second entanglement
        qc.cx(0, 1)
        return qc

    def _build_attention_circuit(self, base_circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Append the self‑attention block to an existing circuit.
        """
        qc = base_circuit.copy()
        # Rotate each qubit with a 3‑parameter rotation
        for i in range(self.n_qubits):
            qc.rx(self.rot_params[3 * i], i)
            qc.ry(self.rot_params[3 * i + 1], i)
            qc.rz(self.rot_params[3 * i + 2], i)
        # Entangle neighbouring qubits with controlled‑RZ gates
        for i in range(self.n_qubits - 1):
            qc.crx(self.ent_params[i], i, i + 1)
        return qc

    def circuit(self) -> QuantumCircuit:
        """
        Construct the full hybrid circuit: sampler -> attention.
        """
        sampler_circuit = self._build_sampler_circuit()
        full_circuit = self._build_attention_circuit(sampler_circuit)
        return full_circuit

    def run(self, shots: int = 1024) -> dict:
        """
        Execute the circuit on a simulator and return measurement counts.

        Parameters
        ----------
        shots: int
            Number of shots for sampling.
        """
        qc = self.circuit()
        job = qiskit.execute(qc, self.backend, shots=shots)
        return job.result().get_counts(qc)


__all__ = ["HybridSamplerAttentionQuantum"]
