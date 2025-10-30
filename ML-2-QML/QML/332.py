"""
Quantum‑classical hybrid estimator extending the original example.

* Feature map: 2‑qubit ZZFeatureMap encoding two classical inputs.
* Ansatz: RealAmplitudes with two layers (one entangling layer per qubit).
* Observable: Pauli‑Z on qubit 0, allowing the expectation value to serve as the regression output.
* The circuit is constructed with Qiskit and wrapped by
  qiskit_machine_learning.neural_networks.EstimatorQNN.

The estimator can be trained with Qiskit's StatevectorEstimator or a backend of your choice.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

# Feature‑map parameters
x1 = Parameter("x1")
x2 = Parameter("x2")

# Weight parameters for the variational ansatz
w1 = Parameter("w1")
w2 = Parameter("w2")
w3 = Parameter("w3")
w4 = Parameter("w4")

def EstimatorQNN() -> QiskitEstimatorQNN:
    """
    Construct a hybrid quantum‑classical regressor.

    Returns
    -------
    qiskit_machine_learning.neural_networks.EstimatorQNN
        A ready‑to‑train Qiskit EstimatorQNN instance.
    """
    # Feature map: encode inputs into a 2‑qubit entangled state
    qc = QuantumCircuit(2)
    qc.ry(x1, 0)
    qc.ry(x2, 1)
    qc.cz(0, 1)

    # Variational ansatz: RealAmplitudes with two layers
    qc.rx(w1, 0)
    qc.rx(w2, 1)
    qc.cz(0, 1)
    qc.rx(w3, 0)
    qc.rx(w4, 1)

    # Observable: Pauli‑Z on qubit 0
    observable = SparsePauliOp.from_list([("Z" + "I", 1)])

    # Wrap into EstimatorQNN
    estimator = StatevectorEstimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[x1, x2],
        weight_params=[w1, w2, w3, w4],
        estimator=estimator,
    )
    return estimator_qnn
