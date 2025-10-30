"""Hybrid quantum‑classical autoencoder using Qiskit.

The quantum part implements a small variational circuit inspired by
the QLayer from the Quantum‑NAT example.  The circuit accepts two sets
of parameters: input features and trainable weights.  A
StatevectorSampler is used to obtain measurement probabilities, which
are fed into a SamplerQNN to produce a deterministic output.

The design integrates ideas from:
  • Qiskit Autoencoder helper
  • SamplerQNN helper
  • QuantumNAT QLayer
"""

import math
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
from typing import Sequence


def _quantum_layer(qc: QuantumCircuit, wires: Sequence[int], params: ParameterVector) -> None:
    """Small variational layer adapted from Quantum‑NAT QLayer."""
    # Random layer: a simple pattern of single‑qubit rotations
    for i, w in zip(wires, params[: len(wires)]):
        qc.ry(w, i)
    # Parameterized single‑qubit gates
    for i, w in zip(wires, params[len(wires) : len(wires) * 2]):
        qc.rx(w, i)
    # Two‑qubit entanglement
    qc.cx(wires[0], wires[1])
    qc.rz(params[len(wires) * 2], wires[0])
    qc.crx(params[len(wires) * 2 + 1], wires[0], wires[1])
    # Static gates
    qc.h(wires[0])
    qc.sx(wires[1])


def build_quantum_autoencoder_circuit(
    num_qubits: int = 4,
    num_weights: int = 10,
) -> QuantumCircuit:
    """Construct a variational circuit that can be used as a SamplerQNN."""
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Input encoding
    inputs = ParameterVector("input", num_qubits)
    for i in range(num_qubits):
        qc.ry(inputs[i], i)

    # Variational layer
    weights = ParameterVector("weight", num_weights)
    _quantum_layer(qc, list(range(num_qubits)), weights)

    # Measurement via a single ancilla for a swap‑test style read‑out
    # Here we simply measure the first qubit in the computational basis
    qc.measure(qr[0], cr[0])
    return qc


def HybridAutoEncoder() -> SamplerQNN:
    """Factory that returns a quantum autoencoder SamplerQNN."""
    algorithm_globals.random_seed = 42
    sampler = StatevectorSampler()

    # Circuit with 4 qubits and 10 trainable weight parameters
    qc = build_quantum_autoencoder_circuit()

    # Define the parameter vectors for input and weight
    input_params = [p for p in qc.parameters if "input" in p.name]
    weight_params = [p for p in qc.parameters if "weight" in p.name]

    def identity_interpret(x):
        return x

    qnn = SamplerQNN(
        circuit=qc,
        input_params=input_params,
        weight_params=weight_params,
        sampler=sampler,
        interpret=identity_interpret,
        output_shape=2,
    )
    return qnn


__all__ = ["HybridAutoEncoder"]
