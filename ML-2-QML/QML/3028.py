"""Quantum helper used by the hybrid estimator.

Provides a one‑qubit parameterised circuit that is wrapped in a
Qiskit `EstimatorQNN`.  The circuit has a single input parameter `x`
and a single weight parameter `w`, and measures the `Y` observable.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

def build_quantum_estimator() -> EstimatorQNN:
    """
    Construct a Qiskit `EstimatorQNN` with a tiny one‑qubit circuit.

    Returns
    -------
    EstimatorQNN
        Quantum neural network ready to be used inside a PyTorch model.
    """
    # Parameterised circuit
    x = Parameter("x")
    w = Parameter("w")
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(x, 0)
    qc.rx(w, 0)

    # Observable to measure
    observable = SparsePauliOp.from_list([("Y", 1)])

    # Primitive estimator (state‑vector simulator)
    estimator = StatevectorEstimator()

    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[x],
        weight_params=[w],
        estimator=estimator,
    )
