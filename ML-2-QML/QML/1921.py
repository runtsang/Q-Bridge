"""
ConvGen108 – Quantum variational filter with parameter‑efficient ansatz.

The QML implementation replaces the single‑shot random circuit with a
depth‑first variational circuit that can be trained via gradient‑based
optimizers.  The circuit is executed on the Aer simulator and outputs
a mean probability of measuring |1⟩ across all qubits.  The design
supports hybrid training with a classical loss that includes the
spectral penalty from the classical counterpart.
"""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter, ParameterVector
from qiskit.opflow import StateFn, CircuitStateFn, PauliExpectation, AerPauliExpectation
from qiskit.quantum_info import Pauli

def ConvGen108(backend: qiskit.providers.Backend = None,
               shots: int = 1000,
               threshold: float = 127,
               n_layers: int = 2,
               kernel_size: int = 2) -> QuantumCircuit:
    """
    Build a depth‑first variational circuit for a kernel of size `kernel_size`.
    Parameters are grouped into a ParameterVector for efficient parameter binding.
    """
    if backend is None:
        backend = Aer.get_backend('qasm_simulator')

    n_qubits = kernel_size ** 2
    theta = ParameterVector('theta', n_qubits * n_layers)

    qc = QuantumCircuit(n_qubits)

    # Encode data into rotation angles
    def encode(data):
        for i, val in enumerate(data):
            angle = np.pi if val > threshold else 0.0
            qc.rz(angle, i)

    # Build layers
    for layer in range(n_layers):
        # Entangling layer
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)
        # Parameterized single‑qubit rotations
        qc.append(qc.rx(theta[layer * n_qubits:(layer + 1) * n_qubits]), range(n_qubits))

    encode_qc = QuantumCircuit(n_qubits)
    encode_qc.decompose(encode)
    qc = encode_qc.compose(qc)

    qc.measure_all()
    qc.save_expectation_value(CircuitStateFn(qc), PauliExpectation(AerPauliExpectation()))
    return qc

def run_convgen108(qc: QuantumCircuit,
                   data: np.ndarray,
                   backend: qiskit.providers.Backend,
                   shots: int = 1000) -> float:
    """
    Execute the variational circuit on classical data and return mean |1⟩ probability.
    """
    data = data.reshape(1, -1)
    param_binds = []
    for dat in data:
        bind = {f'theta{i}': 0.0 for i in range(qc.num_qubits * 2)}
        for i, val in enumerate(dat):
            bind[f'theta{i}'] = np.pi if val > 127 else 0.0
        param_binds.append(bind)

    job = execute(qc, backend, shots=shots, parameter_binds=param_binds)
    result = job.result()
    counts = result.get_counts(qc)
    total = 0
    for key, c in counts.items():
        ones = sum(int(b) for b in key)
        total += ones * c
    mean_prob = total / (shots * qc.num_qubits)
    return mean_prob
