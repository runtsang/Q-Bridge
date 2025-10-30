"""Quantum estimator with a 2‑qubit entangled ansatz and multi‑observable measurement."""
from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit import Aer
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator

def _build_ansatz() -> QuantumCircuit:
    """Constructs a 2‑qubit variational circuit with entanglement layers."""
    qc = QuantumCircuit(2)
    # Input encoding
    inp0, inp1 = Parameter("x0"), Parameter("x1")
    qc.h(0); qc.h(1)
    qc.rz(inp0, 0); qc.rz(inp1, 1)

    # Variational layer 1
    w0, w1, w2, w3 = (Parameter(f"w{i}") for i in range(4))
    qc.cx(0, 1)
    qc.ry(w0, 0); qc.ry(w1, 1)
    qc.cx(1, 0)
    qc.ry(w2, 0); qc.ry(w3, 1)

    # Variational layer 2
    w4, w5, w6, w7 = (Parameter(f"w{i}") for i in range(4, 8))
    qc.cx(0, 1)
    qc.rz(w4, 0); qc.rz(w5, 1)
    qc.cx(1, 0)
    qc.rz(w6, 0); qc.rz(w7, 1)
    return qc

def EstimatorQNN() -> QiskitEstimatorQNN:
    """Instantiate a Qiskit EstimatorQNN with advanced ansatz and observables."""
    qc = _build_ansatz()

    # Observables: Y on qubit 0, Y on qubit 1, and joint YY
    obs_y0 = SparsePauliOp.from_list([("Y0", 1.0)])
    obs_y1 = SparsePauliOp.from_list([("Y1", 1.0)])
    obs_yy = SparsePauliOp.from_list([("YY", 1.0)])
    observables = [obs_y0, obs_y1, obs_yy]

    # Parameter lists
    input_params = [qc.parameters[0], qc.parameters[1]]
    weight_params = qc.parameters[2:]

    # Backend and estimator
    backend = Aer.get_backend("statevector_simulator")
    estimator = QiskitEstimator(backend=backend)

    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
        gradient_method="parameter-shift",
    )

__all__ = ["EstimatorQNN"]
