"""
EstimatorQNN – a variational quantum regression circuit.

The quantum model mirrors the structure of the enriched classical
network.  It uses a 3‑qubit circuit with two layers of parameterised
rotations and CX entanglement, and measures the Pauli‑Z observable on
the first qubit.  The first qubit encodes the input feature, while
the remaining qubits carry trainable weight parameters.
"""

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

def EstimatorQNN() -> QiskitEstimatorQNN:
    """
    Build a variational regression circuit that returns a
    Qiskit EstimatorQNN instance.
    """
    # Input and weight parameters
    x = Parameter("x")   # input feature
    w1 = Parameter("w1") # first weight
    w2 = Parameter("w2") # second weight

    # 3‑qubit circuit
    qc = QuantumCircuit(3)

    # Feature encoding on qubit 0
    qc.ry(x, 0)

    # Two variational layers with entanglement
    for _ in range(2):
        qc.ry(w1, 1)
        qc.rx(w2, 2)
        qc.cx(1, 2)
        qc.cx(0, 1)

    # Observable: Pauli‑Z on qubit 0
    observable = SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)])

    # Construct the EstimatorQNN
    estimator = StatevectorEstimator()
    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[x],
        weight_params=[w1, w2],
        estimator=estimator,
    )

__all__ = ["EstimatorQNN"]
