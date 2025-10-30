"""Quantum circuit builder and executor for the hybrid classifier.

This module implements a parameterized quantum circuit with data‑encoding
via RX rotations, a variational depth of Ry rotations, and CZ entanglement.
It returns expectation values of single‑qubit Z observables that can be fed
into the classical head.

Typical usage::

    from UnifiedClassifierHybrid_qml import build_classifier_circuit, run_circuit
    qc, enc, var, obs = build_classifier_circuit(num_qubits=4, depth=2)
    backend = qiskit.Aer.get_backend('aer_simulator_statevector')
    exps = run_circuit(qc, enc, var, backend)
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.providers import Backend

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """Create a layered variational circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit
        The constructed quantum circuit.
    encoding_params
        ParameterVector for the data‑encoding RX gates.
    variational_params
        ParameterVector for the variational Ry gates.
    observables
        List of SparsePauliOp observables (Z on each qubit).
    """
    encoding = ParameterVector("x", num_qubits)
    var_params = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    # Data‑encoding: RX per qubit
    for qubit, param in enumerate(encoding):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        # Variational Ry rotations
        for qubit in range(num_qubits):
            qc.ry(var_params[idx], qubit)
            idx += 1
        # Entanglement: CZ on adjacent qubits
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return qc, list(encoding), list(var_params), observables

def encode_features(features: np.ndarray, encoding_params: Iterable[ParameterVector]) -> np.ndarray:
    """Map a batch of classical features to values for the encoding parameters.

    The mapping is a simple tanh of the feature values, one per qubit.
    """
    num_qubits = len(encoding_params)
    # Ensure features shape (batch, num_qubits)
    flat = features.reshape(-1, num_qubits)
    return np.tanh(flat)

def run_circuit(
    circuit: QuantumCircuit,
    encoding_params: Iterable[ParameterVector],
    variational_params: Iterable[ParameterVector],
    backend: Backend,
    shots: int = 1024,
) -> np.ndarray:
    """Execute the circuit and return expectation values of the Z observables.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit with unbound parameters.
    encoding_params, variational_params : Iterable[ParameterVector]
        Parameter vectors that will be bound.  For demonstration, all are set to 0.0.
    backend : Backend
        Qiskit backend to run the simulation.
    shots : int
        Number of shots if a state‑vector backend is not used.

    Returns
    -------
    expectations : np.ndarray
        Array of expectation values, one per qubit.
    """
    # Bind all parameters to zero for a deterministic demo
    param_bind = {p: 0.0 for p in encoding_params + variational_params}
    bound_circuit = circuit.bind_parameters(param_bind)

    # Prefer a state‑vector backend for exact expectations
    if "statevector" in backend.name():
        sv = Statevector(bound_circuit)
        expectations = []
        for qubit in range(bound_circuit.num_qubits):
            obs = SparsePauliOp("I"*qubit + "Z" + "I"*(bound_circuit.num_qubits - qubit - 1))
            expectations.append(sv.expectation_value(obs).real)
        return np.array(expectations)

    # Fallback: use a shot simulation and estimate expectations
    bound_circuit.measure_all()
    job = backend.run(qiskit.execute(bound_circuit, backend, shots=shots))
    result = job.result()
    counts = result.get_counts()
    probs = np.array([counts.get(bin(i)[2:].zfill(circuit.num_qubits), 0) for i in range(2**circuit.num_qubits)]) / shots
    expectations = []
    for qubit in range(circuit.num_qubits):
        exp = 0.0
        for state, prob in enumerate(probs):
            bits = format(state, f"0{circuit.num_qubits}b")
            exp += prob * (1 if bits[-(qubit+1)] == "0" else -1)
        expectations.append(exp)
    return np.array(expectations)

__all__ = [
    "build_classifier_circuit",
    "encode_features",
    "run_circuit",
]
