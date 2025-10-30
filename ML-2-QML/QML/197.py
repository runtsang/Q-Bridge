from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator

def EstimatorQNN():
    """
    Variational quantum circuit with 3 qubits, parameterised rotations, and entanglement.
    The circuit is wrapped in Qiskit ML's EstimatorQNN to provide a gradient‑based training interface.
    Multiple Pauli observables are used to enable multi‑output regression.
    """
    # Define parameters
    input_params = ParameterVector("x", 3)
    weight_params = ParameterVector("w", 9)  # 3 qubits × 3 rotations each

    qc = QuantumCircuit(3)

    # Encode inputs
    for i, p in enumerate(input_params):
        qc.ry(p, i)

    # First rotation layer
    for i, w in enumerate(weight_params[0:3]):
        qc.rx(w, i)
    for i, w in enumerate(weight_params[3:6]):
        qc.ry(w, i)
    for i, w in enumerate(weight_params[6:9]):
        qc.rz(w, i)

    # Entanglement
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Second rotation layer (offset to diversify parameters)
    for i, w in enumerate(weight_params[0:3]):
        qc.rx(w + 1.0, i)
    for i, w in enumerate(weight_params[3:6]):
        qc.ry(w + 1.0, i)
    for i, w in enumerate(weight_params[6:9]):
        qc.rz(w + 1.0, i)

    # Observables: Y, Z, and X on each qubit for multi‑output
    obs = [
        SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)]),
        SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)]),
        SparsePauliOp.from_list([("X" * qc.num_qubits, 1)])
    ]

    # Use Aer simulator for statevector estimation
    backend = Aer.get_backend("statevector_simulator")
    estimator = Estimator(backend=backend)

    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=obs,
        input_params=[input_params[i] for i in range(3)],
        weight_params=[weight_params[i] for i in range(9)],
        estimator=estimator,
    )
    return estimator_qnn

__all__ = ["EstimatorQNN"]
