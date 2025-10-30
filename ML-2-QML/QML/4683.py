"""Quantum hybrid estimator mirroring :class:`EstimatorQNNGen`.

The quantum circuit consists of:
  1. A feature‑map using RealAmplitudes.
  2. An attention‑style subcircuit with parameterised rotations and
     controlled‑X gates.
  3. A swap‑test auto‑encoder that projects the encoded state into a
     latent subspace.
  4. A StatevectorEstimator that returns the expectation value of a
     Pauli‑Y observable, giving a scalar output.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.primitives import StatevectorEstimator


def _build_feature_map(num_qubits: int, reps: int = 3) -> QuantumCircuit:
    """RealAmplitudes feature map."""
    return RealAmplitudes(num_qubits, reps=reps)


def _build_attention_subcircuit(num_qubits: int, params: list[Parameter]) -> QuantumCircuit:
    """Quantum self‑attention block."""
    qc = QuantumCircuit(num_qubits)
    # Parameterised single‑qubit rotations
    for i in range(num_qubits):
        qc.rx(params[3 * i], i)
        qc.ry(params[3 * i + 1], i)
        qc.rz(params[3 * i + 2], i)
    # Entangling controlled‑X gates
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def _build_autoencoder_circuit(num_latent: int, num_trash: int, params: list[Parameter]) -> QuantumCircuit:
    """Swap‑test auto‑encoder that projects to a latent subspace."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode with RealAmplitudes and attach parameters
    qc.compose(RealAmplitudes(num_latent + num_trash, reps=2), range(0, num_latent + num_trash), inplace=True)

    # Swap‑test using one auxiliary qubit
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    # Append user‑defined parameters (e.g., attention weights)
    for i, p in enumerate(params):
        qc.rz(p, (i % qc.num_qubits))
    return qc


def EstimatorQNNGenQML() -> EstimatorQNN:
    """
    Returns a quantum variational estimator that mirrors the classical
    :class:`EstimatorQNNGen` architecture.
    """
    # Define qubit counts
    num_qubits = 6          # feature‑map + attention + auto‑encoder
    num_latent = 3
    num_trash = 2

    # Parameters
    feature_params = [Parameter(f"f{i}") for i in range(num_qubits)]
    attention_params = [Parameter(f"a{i}") for i in range(3 * num_qubits)]
    autoencoder_params = [Parameter(f"ae{i}") for i in range(10)]  # arbitrary count

    # Build sub‑circuits
    feature_circuit = _build_feature_map(num_qubits)
    attention_circuit = _build_attention_subcircuit(num_qubits, attention_params)
    autoencoder_circuit = _build_autoencoder_circuit(num_latent, num_trash, autoencoder_params)

    # Assemble full circuit
    full_circuit = QuantumCircuit(num_qubits)
    full_circuit.compose(feature_circuit, inplace=True)
    full_circuit.compose(attention_circuit, inplace=True)
    full_circuit.compose(autoencoder_circuit, inplace=True)

    # Observable for a scalar output
    observable = np.array(["Y"] * num_qubits)

    # Estimator
    estimator = StatevectorEstimator(Aer.get_backend("statevector_simulator"))

    # Wrap into EstimatorQNN
    qnn = EstimatorQNN(
        circuit=full_circuit,
        observables=observable,
        input_params=feature_params,
        weight_params=attention_params + autoencoder_params,
        estimator=estimator,
    )
    return qnn


__all__ = ["EstimatorQNNGenQML"]
