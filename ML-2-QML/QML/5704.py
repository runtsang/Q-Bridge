from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.circuit.library import RealAmplitudes

def EstimatorQNN(num_qubits: int = 2,
                 entanglement: str = "circular") -> QiskitEstimatorQNN:
    """
    Build a hybrid quantum‑classical neural network with a variational circuit.
    Parameters:
        num_qubits: number of qubits in the circuit.
        entanglement: entanglement pattern ("full", "circular", "linear").
    Returns:
        QiskitEstimatorQNN instance.
    """
    # Input parameters
    input_params = [Parameter(f"input_{i}") for i in range(num_qubits)]
    # Weight parameters for the ansatz
    weight_params = [Parameter(f"theta_{i}") for i in range(num_qubits * 2)]

    # Build circuit
    qc = QuantumCircuit(num_qubits)
    # Apply input rotations
    for i, param in enumerate(input_params):
        qc.ry(param, i)

    # Add ansatz layers
    ansatz = RealAmplitudes(num_qubits,
                            entanglement=entanglement,
                            reps=2,
                            insert_barriers=True)
    ansatz.assign_parameters({p: weight_params[idx] for idx, p in enumerate(ansatz.parameters)}, inplace=True)
    qc.append(ansatz, qc.qubits)

    # Measurement observable: sum of Z on all qubits
    observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])

    # Estimator
    estimator = StatevectorEstimator()

    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
