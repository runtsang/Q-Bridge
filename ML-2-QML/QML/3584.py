"""Quantum‑centric EstimatorQNN that encodes inputs with angle rotations
and applies a variational layer inspired by QuantumNAT.

The circuit consists of:
1.  Angle‑encoding of the first input feature on qubit 0 via RY.
2.  A trainable RX rotation on qubit 1 representing a weight parameter.
3.  A variational layer of parameterised RX/RY/RZ gates on all qubits,
   followed by linear‑chain CNOT entanglement.
4.  Additional single‑qubit gates (H, SX, CNOT) borrowed from the
   Quantum‑NAT implementation.
The observable is a global Z‑basis measurement on all qubits.
The returned EstimatorQNN instance can be used directly with
PyTorch or other classical optimisers via the Qiskit
Machine‑Learning primitives.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as QiskitEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN

def EstimatorQNN() -> QiskitEstimatorQNN:
    """Return a parameterised quantum neural network."""
    n_qubits = 4
    input_dim = 2
    weight_dim = 1  # one trainable weight per circuit (RX on qubit 1)
    total_params = input_dim + weight_dim
    params = ParameterVector("theta", total_params)
    input_params = params[:input_dim]
    weight_params = params[input_dim:]

    qc = QuantumCircuit(n_qubits)

    # 1. Angle‑encoding of the first input feature
    qc.ry(input_params[0], 0)

    # 2. Trainable weight rotation on qubit 1
    qc.rx(weight_params[0], 1)

    # 3. Variational layer (parameterised rotations on every qubit)
    for i in range(n_qubits):
        qc.rx(Parameter(f"rx_{i}"), i)
        qc.ry(Parameter(f"ry_{i}"), i)
        qc.rz(Parameter(f"rz_{i}"), i)

    # Entanglement: linear chain of CNOTs
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    # 4. Additional gates inspired by Quantum‑NAT
    qc.h(n_qubits - 1)          # Hadamard on the last qubit
    qc.sx(n_qubits - 2)         # SX on the second‑last qubit
    qc.cx(n_qubits - 1, 0)      # CNOT from last to first qubit

    # Measurement observable: global Z
    observable = SparsePauliOp.from_list([("Z" * n_qubits, 1)])

    # Create the estimator and the QNN instance
    estimator = QiskitEstimator()
    qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=list(input_params),
        weight_params=list(weight_params),
        estimator=estimator,
    )
    return qnn

__all__ = ["EstimatorQNN"]
