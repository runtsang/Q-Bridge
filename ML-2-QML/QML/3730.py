import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info import SparsePauliOp

def _auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Swap‑test based quantum auto‑encoder from the seed."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # ansatz
    def ansatz(num_qubits):
        return RealAmplitudes(num_qubits, reps=5)

    circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()
    auxiliary_qubit = num_latent + 2 * num_trash
    circuit.h(auxiliary_qubit)
    for i in range(num_trash):
        circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
    circuit.h(auxiliary_qubit)
    circuit.measure(auxiliary_qubit, cr[0])
    return circuit

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """QCNN convolutional layer."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[i], 0)
        sub.ry(params[i + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[i + 2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi / 2, 0)
        qc.append(sub.to_instruction(), [i, i + 1])
    return qc

def _pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """QCNN pooling layer."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[i], 0)
        sub.ry(params[i + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[i + 2], 1)
        qc.append(sub.to_instruction(), [i, i + 1])
    return qc

def HybridAutoencoderQML(
    num_latent: int = 3,
    num_trash: int = 2,
    num_features: int = 8,
) -> SamplerQNN:
    """Hybrid quantum auto‑encoder that stitches QCNN ansatz with a swap‑test."""
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    # QCNN feature map + ansatz
    feature_map = ZFeatureMap(num_features)
    conv = _conv_layer(num_features, "c1")
    pool = _pool_layer(num_features, "p1")

    qnn_circuit = QuantumCircuit(num_features)
    qnn_circuit.compose(feature_map, inplace=True)
    qnn_circuit.compose(conv, inplace=True)
    qnn_circuit.compose(pool, inplace=True)

    # Auto‑encoder swap‑test block
    ae_circuit = _auto_encoder_circuit(num_latent, num_trash)

    # Concatenate circuits: feature map + ansatz + auto‑encoder
    circuit = qnn_circuit
    circuit.compose(ae_circuit, inplace=True)

    # QNN construction
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

__all__ = ["HybridAutoencoderQML"]
