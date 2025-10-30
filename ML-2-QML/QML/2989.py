import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# ------------------------------------------------------------
# Quantum QCNN building blocks
# ------------------------------------------------------------
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unitary."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi/2, 0)
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unitary."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(f"{prefix}_conv", length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.append(conv_circuit(params[i*3:i*3+3]), [i, i+1])
    return qc

def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(f"{prefix}_pool", length=(num_qubits//2) * 3)
    for i in range(0, num_qubits, 2):
        qc.append(pool_circuit(params[(i//2)*3:(i//2)*3+3]), [i, i+1])
    return qc

# ------------------------------------------------------------
# Quantum self‑attention block (variational circuit)
# ------------------------------------------------------------
def attention_block(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Self‑attention style entanglement using CRX gates."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(f"{prefix}_attn", length=num_qubits-1)
    for i in range(num_qubits-1):
        qc.crx(params[i], i, i+1)
    return qc

# ------------------------------------------------------------
# Full hybrid quantum circuit
# ------------------------------------------------------------
def QuantumSelfAttentionQCNN(num_qubits: int = 8, embed_dim: int = 4) -> EstimatorQNN:
    """
    Constructs a QCNN followed by a self‑attention block.
    Returns an EstimatorQNN ready for training or inference.
    """
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map (ZFeatureMap) – omitted for brevity; assume identity
    feature_map = QuantumCircuit(num_qubits)

    # QCNN layers
    qc = QuantumCircuit(num_qubits)
    qc.compose(conv_layer(num_qubits, "c1"), list(range(num_qubits)), inplace=True)
    qc.compose(pool_layer(num_qubits, "p1"), list(range(num_qubits)), inplace=True)
    qc.compose(conv_layer(num_qubits//2, "c2"), list(range(num_qubits//2, num_qubits)), inplace=True)
    qc.compose(pool_layer(num_qubits//2, "p2"), list(range(num_qubits//2, num_qubits)), inplace=True)

    # Self‑attention block
    qc.compose(attention_block(num_qubits, "attn"), list(range(num_qubits)), inplace=True)

    # Observable: single‑qubit Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits-1), 1)])

    qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=qc.parameters,
        estimator=estimator,
    )
    return qnn

def SelfAttention() -> EstimatorQNN:
    """Factory returning the hybrid quantum self‑attention model."""
    return QuantumSelfAttentionQCNN()

__all__ = ["SelfAttention", "QuantumSelfAttentionQCNN"]
