"""Quantum QCNNGen299 model with enhanced ansatz and noise modelling."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.providers.basicaer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, amplitude_damping_error

# ----------------------------------------------------------------------
# Helper circuits for convolution and pooling
# ----------------------------------------------------------------------
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unitary with three variational angles."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unitary with three variational angles."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Wraps conv_circuit into a full convolutional layer."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc


def pool_layer(sources, sinks, param_prefix: str) -> QuantumCircuit:
    """Wraps pool_circuit into a full pooling layer."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, sink in zip(sources, sinks):
        qc.append(pool_circuit(params[param_index:param_index + 3]), [src, sink])
        qc.barrier()
        param_index += 3
    return qc


# ----------------------------------------------------------------------
# Full QCNN ansatz construction
# ----------------------------------------------------------------------
def QCNNGen299QNN() -> EstimatorQNN:
    """Builds a QCNN ansatz with noise modelling and multi‑observable readout."""
    # Set global random seed for reproducibility
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map – two‑layer Z‑feature map for 8 qubits
    feature_map = ZFeatureMap(8, reps=2, entanglement="circular")

    # Ansatz definition
    ansatz = QuantumCircuit(8, name="Ansatz")

    # Layer 1: convolution + pooling
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Layer 2: convolution + pooling
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Layer 3: convolution + pooling
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable – single‑qubit Z measurement on the last qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Optional noise model
    noise_model = NoiseModel()
    amp_err = amplitude_damping_error(0.01)
    for qubit in range(8):
        noise_model.add_quantum_error(amp_err, "id", [qubit])

    # Build the EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
        noise_model=noise_model
    )
    return qnn


__all__ = ["QCNNGen299QNN"]
